# ============================================================================
# StudentIO Training Loop
# ============================================================================
#
# This module implements the end-to-end training loop for StudentIO.
#
# Training Pipeline:
#   1. Sample synthetic students from task distribution
#   2. Collect rollouts using current policy
#   3. Compute rewards and advantages (GAE)
#   4. Update policy with PPO
#   5. Update belief filter with reconstruction loss
#   6. Meta-update (optional, for across-student generalization)
#
# Algorithm: PPO (Proximal Policy Optimization)
#
# Justification:
#   - Stable training with clipped objective
#   - Works well with recurrent policies
#   - Handles hybrid action spaces
#   - Good sample efficiency
#
# ============================================================================

using Flux
using Flux.Optimise: update!
using Statistics
using Random
using LinearAlgebra
using Printf

"""
    PPOConfig

Configuration for PPO training.

# Fields
- `clip_ε::Float32`: Clipping parameter (default: 0.2)
- `γ::Float32`: Discount factor (default: 0.99)
- `λ::Float32`: GAE parameter (default: 0.95)
- `value_coef::Float32`: Value loss weight (default: 0.5)
- `entropy_coef::Float32`: Entropy bonus (default: 0.01)
- `epochs_per_update::Int`: PPO epochs per update (default: 4)
- `batch_size::Int`: Minibatch size (default: 64)
- `max_grad_norm::Float32`: Gradient clipping (default: 0.5)
- `learning_rate::Float32`: Adam learning rate (default: 3e-4)
- `reconstruction_coef::Float32`: Belief filter reconstruction loss weight
"""
struct PPOConfig
    clip_ε::Float32
    γ::Float32
    λ::Float32
    value_coef::Float32
    entropy_coef::Float32
    epochs_per_update::Int
    batch_size::Int
    max_grad_norm::Float32
    learning_rate::Float32
    reconstruction_coef::Float32
    
    function PPOConfig(;
        clip_ε::Float32 = 0.2f0,
        γ::Float32 = 0.99f0,
        λ::Float32 = 0.95f0,
        value_coef::Float32 = 0.5f0,
        entropy_coef::Float32 = 0.01f0,
        epochs_per_update::Int = 4,
        batch_size::Int = 64,
        max_grad_norm::Float32 = 0.5f0,
        learning_rate::Float32 = 3.0f-4,
        reconstruction_coef::Float32 = 0.1f0
    )
        @assert clip_ε > 0 && clip_ε < 1 "clip_ε must be in (0, 1)"
        @assert γ > 0 && γ <= 1 "γ must be in (0, 1]"
        @assert λ > 0 && λ <= 1 "λ must be in (0, 1]"
        new(clip_ε, γ, λ, value_coef, entropy_coef, epochs_per_update,
            batch_size, max_grad_norm, learning_rate, reconstruction_coef)
    end
end

"""
    TrainingState

Mutable state for training loop.

# Fields
- `model`: StudentIO model
- `optimizer`: Optimizer state
- `config::PPOConfig`: Training configuration
- `episode_count::Int`: Total episodes completed
- `update_count::Int`: Total parameter updates
- `metrics::Vector{NamedTuple}`: Training metrics history
"""
mutable struct TrainingState{T}
    model::StudentIOModel{T}
    optimizer::Any
    config::PPOConfig
    episode_count::Int
    update_count::Int
    metrics::Vector{NamedTuple}
    reward_log::RewardLog
end

"""
    TrainingState(model, config) -> TrainingState

Initialize training state.
"""
function TrainingState(model::StudentIOModel{T}, config::PPOConfig = PPOConfig()) where T
    optimizer = Flux.Adam(config.learning_rate)
    TrainingState{T}(model, optimizer, config, 0, 0, NamedTuple[], RewardLog())
end

# ============================================================================
# Rollout Collection
# ============================================================================

"""
    RolloutBuffer

Storage for collected trajectories.

# Fields
- `observations::Vector{Vector}`: Observations at each step
- `actions::Vector`: Actions taken
- `rewards::Vector{Float32}`: Rewards received
- `values::Vector{Float32}`: Value estimates
- `log_probs::Vector{Float32}`: Log probabilities of actions
- `belief_states::Vector{Vector}`: Belief states
- `true_states::Vector{Vector}`: Ground truth states (for reconstruction)
- `dones::Vector{Bool}`: Episode termination flags
"""
struct RolloutBuffer{T<:AbstractFloat}
    observations::Vector{Vector{T}}
    actions::Vector{NamedTuple}
    rewards::Vector{T}
    values::Vector{T}
    log_probs::Vector{T}
    belief_states::Vector{Vector{T}}
    true_states::Vector{Vector{T}}
    dones::Vector{Bool}
end

RolloutBuffer{T}() where T = RolloutBuffer{T}(
    Vector{T}[], NamedTuple[], T[], T[], T[], Vector{T}[], Vector{T}[], Bool[]
)

Base.length(buffer::RolloutBuffer) = length(buffer.observations)

"""
    collect_rollout(model, student, steps) -> RolloutBuffer

Collect a rollout from a synthetic student.

# Arguments
- `model`: StudentIO model
- `student::SyntheticStudent`: Synthetic student
- `steps::Int`: Number of steps to collect

# Returns
- `buffer::RolloutBuffer`: Collected trajectory data
"""
function collect_rollout(model::StudentIOModel{T}, student::SyntheticStudent{T}, 
                         steps::Int) where T
    reset!(student)
    buffer = RolloutBuffer{T}()
    
    config = model.config
    h = zeros(T, config.belief_dim)
    u_prev = zeros(T, config.action_dim)
    
    for t in 1:steps
        # Select action
        action, log_prob = select_action(model.policy, h)
        
        # Get value estimate
        value = value_estimate(model.policy, h)
        
        # Step environment
        observation = step!(student, action)
        
        # Compute reward (using ground truth for synthetic students)
        x_prev = length(student.history) > 1 ? 
                 student.history[end-1].next_state : student.task.prior_knowledge
        x_curr = student.true_state
        
        reward, _ = compute_reward(
            model.reward, x_prev, x_curr, action;
            mastery_dim = config.mastery_dim,
            misconception_dim = config.misconception_dim
        )
        
        # Store in buffer
        push!(buffer.observations, observation)
        push!(buffer.actions, action)
        push!(buffer.rewards, reward)
        push!(buffer.values, value)
        push!(buffer.log_probs, log_prob)
        push!(buffer.belief_states, copy(h))
        push!(buffer.true_states, copy(x_curr))
        push!(buffer.dones, t == steps)
        
        # Update belief state
        h, _ = update_belief(model.filter, h, observation, u_prev)
        u_prev = encode_action(model.policy, action)
    end
    
    return buffer
end

"""
    collect_batch_rollouts(model, task_dist, n_episodes, steps_per_episode)

Collect multiple rollouts in parallel (conceptually).
"""
function collect_batch_rollouts(model::StudentIOModel{T}, 
                                task_dist::TaskDistribution,
                                n_episodes::Int,
                                steps_per_episode::Int) where T
    buffers = RolloutBuffer{T}[]
    config = StudentStateConfig(
        state_dim = model.config.state_dim,
        mastery_dim = model.config.mastery_dim,
        misconception_dim = model.config.misconception_dim,
        abstraction_dim = model.config.abstraction_dim,
        action_dim = model.config.action_dim,
        observation_dim = model.config.observation_dim,
        belief_dim = model.config.belief_dim
    )
    
    for _ in 1:n_episodes
        task = sample_student(task_dist)
        student = SyntheticStudent(task, config)
        buffer = collect_rollout(model, student, steps_per_episode)
        push!(buffers, buffer)
    end
    
    return buffers
end

# ============================================================================
# PPO Loss Computation
# ============================================================================

"""
    compute_ppo_loss(model, buffer, advantages, returns, config) -> loss

Compute the PPO loss for a minibatch.

Loss = -L_clip + c_v * L_value - c_e * H[π]

Where:
- L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
- L_value = (V(s) - G)²
- H[π] = entropy of action distribution
"""
function compute_ppo_loss(model::StudentIOModel{T}, 
                          buffer::RolloutBuffer{T},
                          advantages::Vector{T},
                          returns::Vector{T},
                          config::PPOConfig) where T
    n = length(buffer)
    
    # Initialize accumulators
    policy_loss = zero(T)
    value_loss = zero(T)
    entropy_loss = zero(T)
    reconstruction_loss = zero(T)
    
    for i in 1:n
        h = buffer.belief_states[i]
        action = buffer.actions[i]
        old_log_prob = buffer.log_probs[i]
        advantage = advantages[i]
        return_target = returns[i]
        true_state = buffer.true_states[i]
        
        # New log probability and value
        new_log_prob = log_prob(model.policy, h, action)
        new_value = value_estimate(model.policy, h)
        
        # Policy loss with clipping
        ratio = exp(new_log_prob - old_log_prob)
        clipped_ratio = clamp(ratio, 1 - config.clip_ε, 1 + config.clip_ε)
        policy_loss += -min(ratio * advantage, clipped_ratio * advantage)
        
        # Value loss
        value_loss += (new_value - return_target)^2
        
        # Entropy bonus
        entropy_loss += -action_entropy(model.policy, h)
        
        # Reconstruction loss (belief filter training)
        x_hat = decode_state(model.filter, h)
        reconstruction_loss += sum((x_hat .- true_state).^2)
    end
    
    # Average and combine
    n_inv = T(1.0) / T(n)
    total_loss = policy_loss * n_inv +
                 config.value_coef * value_loss * n_inv +
                 config.entropy_coef * entropy_loss * n_inv +
                 config.reconstruction_coef * reconstruction_loss * n_inv
    
    return total_loss, (
        policy = policy_loss * n_inv,
        value = value_loss * n_inv,
        entropy = -entropy_loss * n_inv,
        reconstruction = reconstruction_loss * n_inv
    )
end

# ============================================================================
# PPO Update Step
# ============================================================================

"""
    ppo_update!(state::TrainingState, buffers::Vector{RolloutBuffer})

Perform PPO update on collected rollouts.
"""
function ppo_update!(state::TrainingState{T}, 
                     buffers::Vector{RolloutBuffer{T}}) where T
    config = state.config
    model = state.model
    
    # Flatten buffers and compute advantages
    all_observations = Vector{T}[]
    all_actions = NamedTuple[]
    all_rewards = T[]
    all_values = T[]
    all_log_probs = T[]
    all_belief_states = Vector{T}[]
    all_true_states = Vector{T}[]
    all_advantages = T[]
    all_returns = T[]
    
    for buffer in buffers
        # Compute GAE for this episode
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            config.γ,
            config.λ
        )
        
        append!(all_observations, buffer.observations)
        append!(all_actions, buffer.actions)
        append!(all_rewards, buffer.rewards)
        append!(all_values, buffer.values)
        append!(all_log_probs, buffer.log_probs)
        append!(all_belief_states, buffer.belief_states)
        append!(all_true_states, buffer.true_states)
        append!(all_advantages, advantages)
        append!(all_returns, returns)
    end
    
    # Normalize advantages
    adv_mean = mean(all_advantages)
    adv_std = std(all_advantages) + T(1e-8)
    all_advantages = (all_advantages .- adv_mean) ./ adv_std
    
    # Create combined buffer for minibatch sampling
    n_samples = length(all_observations)
    combined_buffer = RolloutBuffer{T}(
        all_observations, all_actions, all_rewards, all_values,
        all_log_probs, all_belief_states, all_true_states,
        fill(false, n_samples)
    )
    
    # PPO epochs
    ps = Flux.params(model)
    
    for epoch in 1:config.epochs_per_update
        # Shuffle indices
        indices = shuffle(1:n_samples)
        
        # Minibatch updates
        for batch_start in 1:config.batch_size:n_samples
            batch_end = min(batch_start + config.batch_size - 1, n_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Extract minibatch
            mini_buffer = RolloutBuffer{T}(
                combined_buffer.observations[batch_indices],
                combined_buffer.actions[batch_indices],
                combined_buffer.rewards[batch_indices],
                combined_buffer.values[batch_indices],
                combined_buffer.log_probs[batch_indices],
                combined_buffer.belief_states[batch_indices],
                combined_buffer.true_states[batch_indices],
                combined_buffer.dones[batch_indices]
            )
            
            mini_advantages = all_advantages[batch_indices]
            mini_returns = all_returns[batch_indices]
            
            # Compute gradients
            loss, components = compute_ppo_loss(
                model, mini_buffer, mini_advantages, mini_returns, config
            )
            
            grads = Flux.gradient(ps) do
                l, _ = compute_ppo_loss(
                    model, mini_buffer, mini_advantages, mini_returns, config
                )
                l
            end
            
            # Gradient clipping
            total_norm = zero(T)
            for p in ps
                if grads[p] !== nothing
                    total_norm += sum(grads[p].^2)
                end
            end
            total_norm = sqrt(total_norm)
            
            clip_coef = config.max_grad_norm / (total_norm + T(1e-6))
            if clip_coef < 1
                for p in ps
                    if grads[p] !== nothing
                        grads[p] .*= clip_coef
                    end
                end
            end
            
            # Apply update
            Flux.Optimise.update!(state.optimizer, ps, grads)
        end
    end
    
    state.update_count += 1
    
    # Log metrics
    avg_reward = mean(all_rewards)
    avg_value = mean(all_values)
    
    push!(state.metrics, (
        update = state.update_count,
        avg_reward = avg_reward,
        avg_value = avg_value,
        n_samples = n_samples
    ))
    
    return avg_reward
end

# ============================================================================
# Main Training Loop
# ============================================================================

"""
    train!(model, task_dist; kwargs...) -> training_history

Train StudentIO model on synthetic students.

# Arguments
- `model::StudentIOModel`: Model to train
- `task_dist::TaskDistribution`: Distribution over student tasks

# Keyword Arguments
- `num_episodes::Int=10000`: Total training episodes
- `steps_per_episode::Int=100`: Steps per episode
- `episodes_per_update::Int=16`: Episodes to collect before updating
- `config::PPOConfig=PPOConfig()`: PPO configuration
- `log_interval::Int=100`: Logging frequency
- `save_interval::Int=1000`: Model saving frequency
- `save_path::Union{String,Nothing}=nothing`: Path to save checkpoints

# Returns
- `history::Vector{NamedTuple}`: Training metrics
"""
function train!(model::StudentIOModel{T}, task_dist::TaskDistribution;
                num_episodes::Int = 10000,
                steps_per_episode::Int = 100,
                episodes_per_update::Int = 16,
                config::PPOConfig = PPOConfig(),
                log_interval::Int = 100,
                save_interval::Int = 1000,
                save_path::Union{String, Nothing} = nothing) where T
    
    state = TrainingState(model, config)
    
    @info "Starting training" num_episodes=num_episodes steps_per_episode=steps_per_episode
    
    episodes_collected = 0
    
    while episodes_collected < num_episodes
        # Collect rollouts
        buffers = collect_batch_rollouts(
            model, task_dist, episodes_per_update, steps_per_episode
        )
        episodes_collected += episodes_per_update
        state.episode_count = episodes_collected
        
        # PPO update
        avg_reward = ppo_update!(state, buffers)
        
        # Logging
        if episodes_collected % log_interval == 0
            recent_rewards = [m.avg_reward for m in state.metrics[max(1, end-9):end]]
            @info "Training progress" episode=episodes_collected avg_reward=mean(recent_rewards)
        end
        
        # Checkpointing
        if !isnothing(save_path) && episodes_collected % save_interval == 0
            checkpoint_path = "$(save_path)_ep$(episodes_collected).jld2"
            # save_model(model, checkpoint_path)  # Implement as needed
            @info "Saved checkpoint" path=checkpoint_path
        end
    end
    
    @info "Training complete" total_episodes=episodes_collected updates=state.update_count
    
    return state.metrics
end

# ============================================================================
# Training with Meta-Learning
# ============================================================================

"""
    train_with_meta!(model, task_dist; kwargs...)

Train with both PPO and meta-learning objectives.

Alternates between:
1. Standard PPO updates (within-student optimization)
2. Meta-learning updates (across-student generalization)
"""
function train_with_meta!(model::StudentIOModel{T}, task_dist::TaskDistribution;
                          num_episodes::Int = 10000,
                          steps_per_episode::Int = 100,
                          episodes_per_update::Int = 16,
                          meta_update_interval::Int = 100,
                          ppo_config::PPOConfig = PPOConfig(),
                          meta_config::MetaLearnerConfig = MetaLearnerConfig()) where T
    
    state = TrainingState(model, ppo_config)
    meta_learner = MetaLearner{T}(meta_config)
    
    @info "Starting meta+PPO training"
    
    episodes_collected = 0
    config = StudentStateConfig(
        state_dim = model.config.state_dim,
        mastery_dim = model.config.mastery_dim,
        misconception_dim = model.config.misconception_dim,
        abstraction_dim = model.config.abstraction_dim,
        action_dim = model.config.action_dim,
        observation_dim = model.config.observation_dim,
        belief_dim = model.config.belief_dim
    )
    
    # Episode generator for meta-learning
    function generate_episode_fn(task::StudentTask, steps::Int)
        generate_episode(task, steps; config=config)
    end
    
    while episodes_collected < num_episodes
        # Standard PPO update
        buffers = collect_batch_rollouts(
            model, task_dist, episodes_per_update, steps_per_episode
        )
        episodes_collected += episodes_per_update
        state.episode_count = episodes_collected
        
        ppo_update!(state, buffers)
        
        # Meta-learning update (less frequent)
        if episodes_collected % meta_update_interval == 0
            meta_loss = meta_train_step!(
                meta_learner, model, task_dist, generate_episode_fn
            )
            @info "Meta update" episode=episodes_collected meta_loss=meta_loss
        end
    end
    
    return (ppo_metrics = state.metrics, meta_metrics = meta_learner.training_history)
end

# ============================================================================
# Training Utilities
# ============================================================================

"""
    warmup_training(model, task_dist; steps=1000)

Quick warmup training with reduced settings.
"""
function warmup_training(model::StudentIOModel{T}, task_dist::TaskDistribution;
                         steps::Int = 1000) where T
    config = PPOConfig(
        epochs_per_update = 2,
        batch_size = 32
    )
    
    train!(model, task_dist;
        num_episodes = steps,
        steps_per_episode = 50,
        episodes_per_update = 8,
        config = config,
        log_interval = 100
    )
end

"""
    evaluate_training(model, task_dist, n_eval_episodes) -> metrics

Evaluate trained model on held-out students.
"""
function evaluate_training(model::StudentIOModel{T}, 
                           task_dist::TaskDistribution,
                           n_eval_episodes::Int = 100) where T
    config = StudentStateConfig(
        state_dim = model.config.state_dim,
        mastery_dim = model.config.mastery_dim,
        misconception_dim = model.config.misconception_dim,
        abstraction_dim = model.config.abstraction_dim,
        action_dim = model.config.action_dim,
        observation_dim = model.config.observation_dim,
        belief_dim = model.config.belief_dim
    )
    
    total_reward = zero(T)
    total_learning = zero(T)
    
    for _ in 1:n_eval_episodes
        task = sample_student(task_dist)
        student = SyntheticStudent(task, config)
        buffer = collect_rollout(model, student, 100)
        
        total_reward += sum(buffer.rewards)
        total_learning += compute_actual_learning(student)
    end
    
    (
        mean_episode_reward = total_reward / n_eval_episodes,
        mean_learning_gain = total_learning / n_eval_episodes
    )
end
