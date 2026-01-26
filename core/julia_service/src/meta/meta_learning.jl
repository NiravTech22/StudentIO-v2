# ============================================================================
# StudentIO Meta-Learning Module
# ============================================================================
#
# This module implements meta-learning for fast adaptation to new students.
#
# Core Idea:
#   - Each student is a "task" with unique learning dynamics
#   - Meta-parameters θ (neural network weights) are optimized across students
#   - Fast adaptation happens through recurrent state, not weight updates
#   - New students adapt quickly using learned initialization
#
# Algorithm: First-Order MAML (FOMAML)
#
# Justification:
#   - FOMAML is computationally tractable (no second derivatives)
#   - Provides explicit separation of meta-parameters vs task-specific state
#   - Well-studied convergence properties
#   - Works well with recurrent models
#
# ============================================================================

using Flux
using Flux.Optimise: update!
using Statistics
using Random

"""
    MetaLearnerConfig

Configuration for the meta-learning algorithm.

# Fields
- `inner_lr::Float32`: Learning rate for inner loop (fast adaptation)
- `outer_lr::Float32`: Learning rate for meta-parameter update
- `inner_steps::Int`: Number of adaptation steps per task
- `meta_batch_size::Int`: Number of tasks per meta-update
- `support_steps::Int`: Steps for support set (adaptation)
- `query_steps::Int`: Steps for query set (evaluation)
- `first_order::Bool`: Use first-order approximation (FOMAML)
"""
struct MetaLearnerConfig
    inner_lr::Float32
    outer_lr::Float32
    inner_steps::Int
    meta_batch_size::Int
    support_steps::Int
    query_steps::Int
    first_order::Bool
    
    function MetaLearnerConfig(;
        inner_lr::Float32 = 0.01f0,
        outer_lr::Float32 = 0.001f0,
        inner_steps::Int = 5,
        meta_batch_size::Int = 8,
        support_steps::Int = 20,
        query_steps::Int = 10,
        first_order::Bool = true
    )
        @assert inner_lr > 0 "Inner learning rate must be positive"
        @assert outer_lr > 0 "Outer learning rate must be positive"
        @assert inner_steps > 0 "Inner steps must be positive"
        @assert meta_batch_size > 0 "Meta batch size must be positive"
        new(inner_lr, outer_lr, inner_steps, meta_batch_size, 
            support_steps, query_steps, first_order)
    end
end

"""
    MetaLearner{T<:AbstractFloat}

Meta-learning optimizer for across-student adaptation.

Implements FOMAML:
1. Sample batch of student tasks
2. For each task, adapt using support set
3. Evaluate on query set
4. Update meta-parameters based on query loss

# Fields
- `config::MetaLearnerConfig`: Algorithm configuration
- `optimizer::Any`: Outer loop optimizer (Adam, etc.)
- `training_history::Vector{NamedTuple}`: Training metrics
"""
mutable struct MetaLearner{T<:AbstractFloat}
    config::MetaLearnerConfig
    optimizer::Any
    training_history::Vector{NamedTuple}
end

"""
    MetaLearner(config::MetaLearnerConfig) -> MetaLearner

Create meta-learner with given configuration.
"""
function MetaLearner{T}(config::MetaLearnerConfig = MetaLearnerConfig()) where T
    optimizer = Flux.Adam(config.outer_lr)
    MetaLearner{T}(config, optimizer, NamedTuple[])
end

MetaLearner(config::MetaLearnerConfig = MetaLearnerConfig()) = MetaLearner{Float32}(config)

# ============================================================================
# FOMAML Implementation
# ============================================================================

"""
    inner_adapt(model, task, support_data, config) -> adapted_model

Perform inner loop adaptation for a single task.

This is the "fast adaptation" phase where we update model
parameters to fit a specific student's data.

# Arguments
- `model`: The StudentIO model to adapt
- `task::StudentTask`: Student task parameters
- `support_data::Vector`: Support set [(y₁,u₁), ...]
- `config::MetaLearnerConfig`: Configuration

# Returns
- `adapted_model`: Model with adapted parameters
"""
function inner_adapt(model, support_data::Vector, config::MetaLearnerConfig)
    # Clone model for task-specific adaptation
    adapted_params = deepcopy(Flux.params(model))
    
    # Convert to explicit parameter vector for updates
    ps = Flux.params(model)
    
    # Inner loop: gradient descent on support set
    for step in 1:config.inner_steps
        total_loss = 0.0f0
        
        for (y, u, h_prev) in support_data
            # Forward pass through belief filter
            h_new, uncertainty = update_belief(model.filter, h_prev, y, u)
            
            # Loss: negative log-likelihood + reconstruction
            # (simplified for inner loop efficiency)
            loss = -log_likelihood(model.observation, decode_state(model.filter, h_new), y)
            total_loss += loss
        end
        
        avg_loss = total_loss / length(support_data)
        
        # Compute gradients
        grads = Flux.gradient(ps) do
            loss_sum = 0.0f0
            for (y, u, h_prev) in support_data
                h_new, _ = update_belief(model.filter, h_prev, y, u)
                x_hat = decode_state(model.filter, h_new)
                loss_sum += -log_likelihood(model.observation, x_hat, y)
            end
            loss_sum / length(support_data)
        end
        
        # Inner loop update (SGD for simplicity)
        for p in ps
            if grads[p] !== nothing
                p .-= config.inner_lr .* grads[p]
            end
        end
    end
    
    return model
end

"""
    evaluate_query(model, query_data) -> loss

Evaluate adapted model on query set.
"""
function evaluate_query(model, query_data::Vector)
    total_loss = 0.0f0
    
    for (y, u, h_prev) in query_data
        h_new, uncertainty = update_belief(model.filter, h_prev, y, u)
        x_hat = decode_state(model.filter, h_new)
        loss = -log_likelihood(model.observation, x_hat, y)
        total_loss += loss
    end
    
    return total_loss / length(query_data)
end

"""
    meta_train_step!(learner::MetaLearner, model, task_dist, generate_episode_fn)

Perform one meta-training step.

# Algorithm (FOMAML)
```
for each task in meta_batch:
    1. Clone model
    2. Generate support set (adaptation data)
    3. Inner loop: adapt clone on support set
    4. Generate query set (evaluation data)
    5. Compute query loss with adapted model
    6. Accumulate gradients

Meta-update: θ ← θ - α ∇ Σ L_query
```

# Arguments
- `learner::MetaLearner`: Meta-learning state
- `model`: StudentIO model to train
- `task_dist::TaskDistribution`: Distribution over student tasks
- `generate_episode_fn`: Function (task, steps) → [(y, u, x), ...]
"""
function meta_train_step!(learner::MetaLearner{T}, model, task_dist::TaskDistribution,
                          generate_episode_fn::Function) where T
    config = learner.config
    
    # Accumulate gradients across tasks
    meta_loss = zero(T)
    ps = Flux.params(model)
    accumulated_grads = nothing
    
    for task_idx in 1:config.meta_batch_size
        # Sample a student task
        task = sample_student(task_dist)
        
        # Generate support and query episodes
        support_episode = generate_episode_fn(task, config.support_steps)
        query_episode = generate_episode_fn(task, config.query_steps)
        
        # Prepare data (extract y, u, h_prev)
        support_data = prepare_episode_data(model, support_episode)
        query_data = prepare_episode_data(model, query_episode)
        
        # Clone model for this task
        model_copy = deepcopy(model)
        
        # Inner adaptation
        model_copy = inner_adapt(model_copy, support_data, config)
        
        # Query evaluation
        query_loss = evaluate_query(model_copy, query_data)
        meta_loss += query_loss
        
        # FOMAML: Compute gradients of query loss w.r.t. ORIGINAL parameters
        # (first-order approximation: ignore Hessian)
        if config.first_order
            grads = Flux.gradient(ps) do
                # Re-adapt and evaluate in one pass for gradient
                adapted = inner_adapt(deepcopy(model), support_data, config)
                evaluate_query(adapted, query_data)
            end
            
            if isnothing(accumulated_grads)
                accumulated_grads = grads
            else
                for p in ps
                    if grads[p] !== nothing
                        if accumulated_grads[p] === nothing
                            accumulated_grads[p] = grads[p]
                        else
                            accumulated_grads[p] .+= grads[p]
                        end
                    end
                end
            end
        end
    end
    
    # Average gradients
    meta_loss /= config.meta_batch_size
    if !isnothing(accumulated_grads)
        for p in ps
            if accumulated_grads[p] !== nothing
                accumulated_grads[p] ./= config.meta_batch_size
            end
        end
    end
    
    # Meta-update
    if !isnothing(accumulated_grads)
        Flux.Optimise.update!(learner.optimizer, ps, accumulated_grads)
    end
    
    # Record metrics
    push!(learner.training_history, (
        meta_loss = Float32(meta_loss),
        tasks_processed = config.meta_batch_size
    ))
    
    return meta_loss
end

"""
    prepare_episode_data(model, episode) -> Vector{Tuple}

Convert episode to (y, u, h_prev) tuples for training.
"""
function prepare_episode_data(model, episode::Vector)
    hidden_dim = model.config.belief_dim
    T = Float32
    
    data = Tuple[]
    h_prev = zeros(T, hidden_dim)
    u_prev = zeros(T, model.config.action_dim)
    
    for step in episode
        y = step.observation
        h_new, _ = update_belief(model.filter, h_prev, y, u_prev)
        push!(data, (y, u_prev, h_prev))
        
        h_prev = h_new
        u_prev = step.action
    end
    
    return data
end

# ============================================================================
# Meta-Training Loop
# ============================================================================

"""
    meta_train!(learner::MetaLearner, model, task_dist, generate_episode_fn;
                num_iterations=1000, log_interval=100)

Complete meta-training loop.

# Arguments
- `learner::MetaLearner`: Meta-learner state
- `model`: StudentIO model
- `task_dist::TaskDistribution`: Student distribution
- `generate_episode_fn`: Episode generation function
- `num_iterations::Int`: Number of meta-updates
- `log_interval::Int`: Logging frequency

# Returns
- `training_history::Vector`: Full training metrics
"""
function meta_train!(learner::MetaLearner{T}, model, task_dist::TaskDistribution,
                     generate_episode_fn::Function;
                     num_iterations::Int = 1000,
                     log_interval::Int = 100,
                     early_stop_patience::Int = 50) where T
    
    best_loss = T(Inf)
    patience_counter = 0
    
    @info "Starting meta-training for $num_iterations iterations"
    
    for iter in 1:num_iterations
        loss = meta_train_step!(learner, model, task_dist, generate_episode_fn)
        
        # Logging
        if iter % log_interval == 0
            recent_losses = [h.meta_loss for h in learner.training_history[max(1, end-log_interval+1):end]]
            avg_loss = mean(recent_losses)
            
            @info "Meta-training" iteration=iter avg_loss=round(avg_loss, digits=4)
        end
        
        # Early stopping
        if loss < best_loss
            best_loss = loss
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= early_stop_patience
                @info "Early stopping at iteration $iter"
                break
            end
        end
    end
    
    @info "Meta-training complete" final_loss=best_loss
    
    return learner.training_history
end

# ============================================================================
# Fast Adaptation at Deployment
# ============================================================================

"""
    fast_adapt!(model, observations, actions; steps=5, lr=0.01)

Perform fast adaptation to a new student at deployment time.

Unlike meta-training, this only updates the recurrent hidden state,
NOT the model parameters (which remain fixed after meta-training).

# Arguments
- `model`: Meta-trained StudentIO model
- `observations::Vector`: Initial observations from new student
- `actions::Vector`: Actions taken
- `steps::Int`: Number of adaptation iterations (default: 5)
- `lr::Float32`: Adaptation learning rate

# Returns
- `adapted_h::Vector`: Adapted belief state for this student
"""
function fast_adapt(model, observations::Vector, actions::Vector;
                    steps::Int = 5, lr::Float32 = 0.01f0)
    hidden_dim = model.config.belief_dim
    T = Float32
    
    # Initialize belief state
    h = zeros(T, hidden_dim)
    
    # Process observations to build initial belief
    for i in eachindex(observations)
        u_prev = i > 1 ? actions[i-1] : zeros(T, model.config.action_dim)
        h, _ = update_belief(model.filter, h, observations[i], u_prev)
    end
    
    return h
end

# ============================================================================
# Meta-Learning Diagnostics
# ============================================================================

"""
    adaptation_speed(model, task, generate_episode_fn; max_steps=100) -> Int

Measure how many steps needed to reach 90% performance on a task.
"""
function adaptation_speed(model, task::StudentTask, generate_episode_fn::Function;
                          max_steps::Int = 100, threshold::Float32 = 0.9f0)
    hidden_dim = model.config.belief_dim
    T = Float32
    
    h = zeros(T, hidden_dim)
    u_prev = zeros(T, model.config.action_dim)
    
    performances = Float32[]
    
    for step in 1:max_steps
        episode = generate_episode_fn(task, 1)
        if isempty(episode)
            break
        end
        
        y = episode[1].observation
        h, uncertainty = update_belief(model.filter, h, y, u_prev)
        
        # Measure performance (reconstruction accuracy)
        x_true = episode[1].true_state
        x_hat = decode_state(model.filter, h)
        accuracy = 1.0f0 - norm(x_true - x_hat) / (norm(x_true) + 1e-8f0)
        
        push!(performances, accuracy)
        
        # Check if reached threshold
        if accuracy >= threshold
            return step
        end
        
        u_prev = episode[1].action
    end
    
    return max_steps  # Did not reach threshold
end

"""
    meta_learning_summary(learner::MetaLearner) -> NamedTuple

Get summary statistics from meta-training.
"""
function meta_learning_summary(learner::MetaLearner)
    if isempty(learner.training_history)
        return (iterations = 0, final_loss = NaN, best_loss = NaN)
    end
    
    losses = [h.meta_loss for h in learner.training_history]
    
    (
        iterations = length(learner.training_history),
        final_loss = losses[end],
        best_loss = minimum(losses),
        mean_loss = mean(losses),
        loss_std = std(losses)
    )
end

# ============================================================================
# Task-Specific Fine-Tuning (Optional)
# ============================================================================

"""
    finetune_for_task!(model, task, generate_episode_fn;
                       steps=100, lr=0.0001)

Fine-tune model for a specific student (optional, breaks meta-learning structure).

WARNING: This modifies model parameters and should only be used when
a student will have many interactions (not suitable for few-shot).
"""
function finetune_for_task!(model, task::StudentTask, generate_episode_fn::Function;
                            steps::Int = 100, lr::Float32 = 0.0001f0)
    optimizer = Flux.Adam(lr)
    ps = Flux.params(model)
    
    for step in 1:steps
        episode = generate_episode_fn(task, 10)
        data = prepare_episode_data(model, episode)
        
        loss = 0.0f0
        grads = Flux.gradient(ps) do
            total = 0.0f0
            for (y, u, h_prev) in data
                h_new, _ = update_belief(model.filter, h_prev, y, u)
                x_hat = decode_state(model.filter, h_new)
                total += -log_likelihood(model.observation, x_hat, y)
            end
            total / length(data)
        end
        
        Flux.Optimise.update!(optimizer, ps, grads)
    end
    
    return model
end
