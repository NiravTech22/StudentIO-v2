# ============================================================================
# StudentIO Synthetic Student Simulation
# ============================================================================
#
# This module provides parametric synthetic students for training and evaluation.
#
# Purpose:
#   - Ground truth access: True xₜ is known
#   - Controlled experiments: Vary one parameter at a time
#   - Debugging: Compare belief to actual state
#   - Policy verification: Measure actual learning outcomes
#
# Student Profiles:
#   - Fast Learner: High learning rate, low forgetting
#   - Slow Learner: Low learning rate, moderate forgetting
#   - Strategic: Careful responses, high accuracy when learned
#   - Impulsive: Fast responses, noisy accuracy
#   - Guesser: Random responses, ignores knowledge
#   - Fatigued: Degrading performance over session
#
# ============================================================================

using Random
using LinearAlgebra
using Distributions

"""
    SyntheticStudent{T<:AbstractFloat}

A fully simulated student with known ground truth state.

The synthetic student maintains the TRUE latent state xₜ which is
hidden from the learning system but available for evaluation.

# Fields
- `task::StudentTask{T}`: Task parameters defining learning dynamics
- `true_state::Vector{T}`: Ground truth knowledge state xₜ
- `history::Vector{NamedTuple}`: Full trajectory with ground truth
- `step_count::Int`: Current timestep
- `fatigue_level::T`: Accumulated fatigue (0 to 1)
- `config::StudentStateConfig`: State configuration
"""
mutable struct SyntheticStudent{T<:AbstractFloat}
    task::StudentTask{T}
    true_state::Vector{T}
    history::Vector{NamedTuple}
    step_count::Int
    fatigue_level::T
    config::StudentStateConfig
end

"""
    SyntheticStudent(task::StudentTask{T}, config::StudentStateConfig) where T

Create a new synthetic student from task parameters.
"""
function SyntheticStudent(task::StudentTask{T}, config::StudentStateConfig) where T
    # Initialize state from task's prior knowledge
    true_state = copy(task.prior_knowledge)
    
    # Ensure state has correct dimension
    if length(true_state) != config.state_dim
        true_state = vcat(
            true_state,
            zeros(T, max(0, config.state_dim - length(true_state)))
        )[1:config.state_dim]
    end
    
    SyntheticStudent{T}(
        task,
        true_state,
        NamedTuple[],
        0,
        zero(T),
        config
    )
end

"""
    SyntheticStudent(task::StudentTask) -> SyntheticStudent

Create synthetic student with default configuration.
"""
function SyntheticStudent(task::StudentTask{T}) where T
    config = StudentStateConfig(state_dim=length(task.prior_knowledge))
    SyntheticStudent(task, config)
end

# ============================================================================
# State Transition (Ground Truth Dynamics)
# ============================================================================

"""
    transition_state!(student::SyntheticStudent, action) -> x_next

Apply the true state transition based on action.

This is the GROUND TRUTH dynamics, not the learned model.
"""
function transition_state!(student::SyntheticStudent{T}, action) where T
    task = student.task
    x = student.true_state
    config = student.config
    
    # Extract action parameters
    difficulty = haskey(action, :difficulty) ? action.difficulty : T(0.5)
    problem_id = haskey(action, :problem_id) ? action.problem_id : 1
    
    # Determine which topics are affected by action
    affected_topics = get_affected_topics(action, config.mastery_dim)
    
    # === Learning Effect ===
    # Learning rate depends on:
    # 1. Base learning rate
    # 2. Topic affinity
    # 3. Difficulty match
    # 4. Current mastery (harder to improve when high)
    
    Δx_learn = zeros(T, config.state_dim)
    
    for topic in affected_topics
        if topic <= config.mastery_dim
            current_mastery = x[topic]
            affinity = topic <= length(task.topic_affinities) ? 
                       task.topic_affinities[topic] : T(1.0)
            
            # Optimal learning at difficulty slightly above mastery
            difficulty_match = exp(-((difficulty - current_mastery - T(0.15))^2) / T(0.1))
            
            # Learning diminishes at high mastery
            diminishing_factor = T(1.0) - current_mastery^2
            
            # Net learning
            Δx_learn[topic] = task.learning_rate * affinity * 
                              difficulty_match * diminishing_factor * T(0.1)
        end
    end
    
    # === Forgetting ===
    # All topics decay, especially those not practiced
    Δx_forget = zeros(T, config.state_dim)
    for topic in 1:config.mastery_dim
        if !(topic in affected_topics)
            # Exponential decay for unpracticed topics
            Δx_forget[topic] = -task.forgetting_rate * x[topic] * T(0.01)
        end
    end
    
    # === Misconception Dynamics ===
    # Misconceptions strengthen with confusion, weaken with clarity
    for i in 1:config.misconception_dim
        misc_idx = config.mastery_dim + i
        if misc_idx <= length(x)
            # Confusion from difficulty mismatch
            avg_mastery = mean(x[1:min(10, config.mastery_dim)])
            confusion = abs(difficulty - avg_mastery)
            
            if confusion > 0.3
                Δx_learn[misc_idx] += task.learning_rate * confusion * T(0.02)
            else
                Δx_learn[misc_idx] -= T(0.01)  # Misconceptions fade with practice
            end
        end
    end
    
    # === Abstraction Building ===
    # Abstractions increase when multiple related topics are mastered
    abstraction_start = config.mastery_dim + config.misconception_dim + 1
    if abstraction_start <= config.state_dim
        avg_mastery = mean(x[1:config.mastery_dim])
        if avg_mastery > 0.6
            # Build abstractions when base mastery is solid
            n_abstractions = config.state_dim - abstraction_start + 1
            for i in 1:n_abstractions
                idx = abstraction_start + i - 1
                Δx_learn[idx] += T(0.01) * (avg_mastery - T(0.6))
            end
        end
    end
    
    # === Apply Transition ===
    x_next = x .+ Δx_learn .+ Δx_forget
    
    # Add process noise
    σ_w = task.noise_scale * T(0.02)
    w = σ_w .* randn(T, config.state_dim)
    x_next = x_next .+ w
    
    # Clamp to valid ranges
    x_next[1:config.mastery_dim] = clamp.(x_next[1:config.mastery_dim], T(0), T(1))
    x_next[config.mastery_dim+1:config.mastery_dim+config.misconception_dim] = 
        clamp.(x_next[config.mastery_dim+1:config.mastery_dim+config.misconception_dim], T(0), T(1))
    
    # Update state
    student.true_state = x_next
    student.step_count += 1
    
    return x_next
end

"""
    get_affected_topics(action, mastery_dim) -> Vector{Int}

Determine which topics are affected by an action.
"""
function get_affected_topics(action, mastery_dim::Int)
    if haskey(action, :topic_id) && !isnothing(action.topic_id)
        return [action.topic_id]
    elseif haskey(action, :problem_id) && !isnothing(action.problem_id)
        # Map problem to topics (simplified: each problem affects 1-3 topics)
        base_topic = mod(action.problem_id - 1, mastery_dim) + 1
        return [base_topic, mod(base_topic, mastery_dim) + 1]
    else
        return [1]  # Default to first topic
    end
end

# ============================================================================
# Observation Generation
# ============================================================================

"""
    generate_observation(student::SyntheticStudent, action) -> Vector

Generate a noisy observation based on true state and response style.
"""
function generate_observation(student::SyntheticStudent{T}, action) where T
    task = student.task
    x = student.true_state
    config = student.config
    
    # Base observation from true state
    affected_topics = get_affected_topics(action, config.mastery_dim)
    
    # === Correctness ===
    if !isempty(affected_topics)
        avg_relevant_mastery = mean(x[affected_topics])
    else
        avg_relevant_mastery = mean(x[1:config.mastery_dim])
    end
    
    difficulty = haskey(action, :difficulty) ? action.difficulty : T(0.5)
    
    # Probability of correct response
    # Using IRT-like model: P(correct) = sigmoid(mastery - difficulty)
    logit = T(4.0) * (avg_relevant_mastery - difficulty)
    p_correct = sigmoid_scalar(logit)
    
    # Apply response style
    p_correct = apply_response_style(p_correct, task.response_style, student)
    
    # Sample correctness
    correct = rand() < p_correct ? T(1) : T(0)
    
    # === Response Time ===
    # Log-normal: faster when confident
    base_time = T(10.0)  # seconds
    confidence_factor = p_correct
    
    mean_log_time = log(base_time * (T(1.5) - confidence_factor))
    std_log_time = T(0.3)
    log_time = mean_log_time + std_log_time * randn(T)
    response_time = exp(log_time)
    
    # === Confidence ===
    # Student's self-reported confidence
    true_confidence = avg_relevant_mastery
    noise = task.noise_scale * T(0.1) * randn(T)
    reported_confidence = clamp(true_confidence + noise, T(0), T(1))
    
    # === Partial Credit ===
    # For multi-step problems
    partial = if correct > T(0.5)
        T(1.0)
    else
        # Partial credit based on mastery
        clamp(avg_relevant_mastery + T(0.1) * randn(T), T(0), T(0.9))
    end
    
    # Build observation vector
    obs_dim = config.observation_dim
    observation = zeros(T, obs_dim)
    
    if obs_dim >= 1
        observation[1] = correct
    end
    if obs_dim >= 2
        observation[2] = log(response_time + T(0.1))  # Log-normalized
    end
    if obs_dim >= 3
        observation[3] = reported_confidence
    end
    if obs_dim >= 4
        observation[4] = partial
    end
    
    return observation
end

"""
    sigmoid_scalar(x)

Scalar sigmoid for p_correct computation.
"""
sigmoid_scalar(x::T) where T = T(1) / (T(1) + exp(-x))

"""
    apply_response_style(p_correct, style, student) -> adjusted_p

Adjust response probability based on student's response style.
"""
function apply_response_style(p_correct::T, style::ResponseStyle, 
                              student::SyntheticStudent{T}) where T
    
    if style == CAREFUL
        # More accurate when learned, but takes longer (not reflected here)
        return clamp(p_correct + T(0.1), T(0), T(1))
        
    elseif style == IMPULSIVE
        # Noisier responses
        noise = T(0.15) * randn(T)
        return clamp(p_correct + noise, T(0), T(1))
        
    elseif style == GUESSING
        # Often just guesses
        guess_prob = T(0.3)
        if rand() < guess_prob
            return T(0.5)  # Random guess
        else
            return p_correct
        end
        
    elseif style == STRATEGIC
        # Adjusts effort based on difficulty
        # Gives up on things perceived too hard
        if p_correct < T(0.3)
            return T(0.2)  # Gives up
        else
            return clamp(p_correct + T(0.05), T(0), T(1))
        end
        
    elseif style == FATIGUED
        # Performance degrades over time
        fatigue_factor = T(1.0) - student.fatigue_level * T(0.5)
        
        # Increase fatigue
        student.fatigue_level = min(T(1.0), student.fatigue_level + T(0.01))
        
        return clamp(p_correct * fatigue_factor, T(0), T(1))
    end
    
    return p_correct
end

# ============================================================================
# Full Step Interface
# ============================================================================

"""
    step!(student::SyntheticStudent, action) -> observation

Execute one step: transition state and generate observation.

# Arguments
- `student::SyntheticStudent`: The synthetic student
- `action`: Action taken by the system

# Returns
- `observation::Vector`: Observable response

# Side Effects
- Updates student.true_state
- Updates student.history
- Increments step_count
"""
function step!(student::SyntheticStudent{T}, action) where T
    # Record previous state
    x_prev = copy(student.true_state)
    
    # Transition to new state
    x_next = transition_state!(student, action)
    
    # Generate observation
    observation = generate_observation(student, action)
    
    # Record in history
    push!(student.history, (
        step = student.step_count,
        true_state = x_prev,
        next_state = x_next,
        action = action,
        observation = observation
    ))
    
    return observation
end

"""
    reset!(student::SyntheticStudent)

Reset student to initial state.
"""
function reset!(student::SyntheticStudent{T}) where T
    student.true_state = copy(student.task.prior_knowledge)
    
    # Pad to correct dimension if needed
    if length(student.true_state) < student.config.state_dim
        student.true_state = vcat(
            student.true_state,
            zeros(T, student.config.state_dim - length(student.true_state))
        )
    end
    student.true_state = student.true_state[1:student.config.state_dim]
    
    empty!(student.history)
    student.step_count = 0
    student.fatigue_level = zero(T)
    
    return student
end

# ============================================================================
# Episode Generation
# ============================================================================

"""
    generate_episode(student::SyntheticStudent, policy, steps::Int) -> Vector

Generate an episode of interactions.

# Arguments
- `student::SyntheticStudent`: The student
- `policy`: Action selection policy (function h -> action)
- `steps::Int`: Number of steps

# Returns
- `episode::Vector{NamedTuple}`: Episode with (observation, action, true_state)
"""
function generate_episode(student::SyntheticStudent{T}, policy::Function, 
                          steps::Int;
                          initial_belief::Union{Nothing, Vector} = nothing) where T
    reset!(student)
    
    episode = NamedTuple[]
    h = isnothing(initial_belief) ? zeros(T, student.config.belief_dim) : initial_belief
    
    for _ in 1:steps
        # Select action
        action, _ = policy(h)
        
        # Step student
        observation = step!(student, action)
        
        # Record
        push!(episode, (
            observation = observation,
            action = action,
            true_state = copy(student.true_state)
        ))
    end
    
    return episode
end

"""
    generate_episode(task::StudentTask, steps::Int; config=StudentStateConfig()) -> Vector

Generate episode from task (with random policy).
"""
function generate_episode(task::StudentTask{T}, steps::Int;
                          config::StudentStateConfig = StudentStateConfig()) where T
    student = SyntheticStudent(task, config)
    
    episode = NamedTuple[]
    
    for _ in 1:steps
        # Random action
        action = (
            action_type = rand(instances(ActionType)),
            problem_id = rand(1:100),
            topic_id = rand(1:config.mastery_dim),
            difficulty = rand(T),
            pacing = rand(T),
            emphasis = rand(T)
        )
        
        observation = step!(student, action)
        action_vec = zeros(T, config.action_dim)
        action_vec[1] = action.difficulty
        
        push!(episode, (
            observation = observation,
            action = action_vec,
            true_state = copy(student.true_state)
        ))
    end
    
    return episode
end

# ============================================================================
# Preset Student Profiles
# ============================================================================

"""
    create_fast_learner(config::StudentStateConfig) -> SyntheticStudent

Create a fast-learning, good retention student.
"""
function create_fast_learner(config::StudentStateConfig = StudentStateConfig())
    T = Float32
    task = StudentTask{T}(
        T(0.4),              # High learning rate
        T(0.02),             # Low forgetting
        T(0.15),             # Low noise
        zeros(T, config.state_dim),  # No prior knowledge
        CAREFUL,             # Careful style
        100,                 # Long attention span
        ones(T, config.mastery_dim)  # Even affinities
    )
    SyntheticStudent(task, config)
end

"""
    create_slow_learner(config::StudentStateConfig) -> SyntheticStudent

Create a slow-learning student.
"""
function create_slow_learner(config::StudentStateConfig = StudentStateConfig())
    T = Float32
    task = StudentTask{T}(
        T(0.05),             # Low learning rate
        T(0.1),              # High forgetting
        T(0.3),              # Moderate noise
        zeros(T, config.state_dim),
        STRATEGIC,           # Strategic style
        60,                  # Moderate attention
        ones(T, config.mastery_dim)
    )
    SyntheticStudent(task, config)
end

"""
    create_guessing_student(config::StudentStateConfig) -> SyntheticStudent

Create a student who frequently guesses.
"""
function create_guessing_student(config::StudentStateConfig = StudentStateConfig())
    T = Float32
    task = StudentTask{T}(
        T(0.15),             # Moderate learning rate
        T(0.05),             # Moderate forgetting
        T(0.8),              # High noise
        zeros(T, config.state_dim),
        GUESSING,            # Guessing style
        40,                  # Short attention
        ones(T, config.mastery_dim)
    )
    SyntheticStudent(task, config)
end

"""
    create_fatigued_student(config::StudentStateConfig) -> SyntheticStudent

Create a student who fatigues quickly.
"""
function create_fatigued_student(config::StudentStateConfig = StudentStateConfig())
    T = Float32
    task = StudentTask{T}(
        T(0.2),              # Moderate learning rate
        T(0.03),             # Low forgetting
        T(0.2),              # Moderate noise
        zeros(T, config.state_dim),
        FATIGUED,            # Fatigued style
        30,                  # Short attention
        ones(T, config.mastery_dim)
    )
    SyntheticStudent(task, config)
end

# ============================================================================
# Evaluation Utilities
# ============================================================================

"""
    compute_actual_learning(student::SyntheticStudent) -> Real

Compute actual learning that occurred (ground truth).
"""
function compute_actual_learning(student::SyntheticStudent{T}) where T
    if length(student.history) < 2
        return zero(T)
    end
    
    initial_state = student.history[1].true_state
    final_state = student.history[end].next_state
    
    config = student.config
    initial_mastery = mean(initial_state[1:config.mastery_dim])
    final_mastery = mean(final_state[1:config.mastery_dim])
    
    return final_mastery - initial_mastery
end

"""
    get_ground_truth_trajectory(student::SyntheticStudent) -> Matrix

Get full ground truth state trajectory.
"""
function get_ground_truth_trajectory(student::SyntheticStudent{T}) where T
    n = length(student.history)
    if n == 0
        return zeros(T, student.config.state_dim, 0)
    end
    
    trajectory = zeros(T, student.config.state_dim, n)
    for (i, step) in enumerate(student.history)
        trajectory[:, i] = step.next_state
    end
    
    return trajectory
end
