# ============================================================================
# StudentIO Reward System
# ============================================================================
#
# This module implements the learning-centric reward function that guides
# policy optimization toward educationally meaningful objectives.
#
# Mathematical Foundation:
#   R(xₜ, uₜ) = α·R_gain + β·R_retention + γ·R_transfer
#
# Where:
#   - R_gain: Immediate knowledge acquisition
#   - R_retention: Long-term memory preservation
#   - R_transfer: Ability to generalize to new contexts
#
# Design Principles:
#   1. Optimize for LEARNING, not engagement
#   2. Penalize forgetting, not just non-progress
#   3. Reward transfer/generalization as highest goal
#   4. Prevent reward hacking (e.g., easy problems only)
#
# ============================================================================

using Statistics

"""
    RewardConfig

Configuration for reward function weights and thresholds.

# Fields
- `α::Float32`: Weight for knowledge gain (default: 0.5)
- `β::Float32`: Weight for retention (default: 0.3)
- `γ::Float32`: Weight for transfer (default: 0.2)
- `mastery_threshold::Float32`: Threshold for "learned" status (default: 0.7)
- `forgetting_penalty::Float32`: Base penalty for forgetting (default: 0.1)
- `difficulty_bonus_scale::Float32`: Bonus for appropriate difficulty (default: 0.1)
- `efficiency_weight::Float32`: Weight for learning efficiency (default: 0.05)

# Constraints
Weights (α, β, γ) should sum to 1.0 for normalized rewards.
Mastery threshold should be in [0.5, 0.9] for meaningful learning.
"""
struct RewardConfig
    α::Float32  # Knowledge gain weight
    β::Float32  # Retention weight
    γ::Float32  # Transfer weight
    mastery_threshold::Float32
    forgetting_penalty::Float32
    difficulty_bonus_scale::Float32
    efficiency_weight::Float32
    
    function RewardConfig(;
        α::Float32 = 0.5f0,
        β::Float32 = 0.3f0,
        γ::Float32 = 0.2f0,
        mastery_threshold::Float32 = 0.7f0,
        forgetting_penalty::Float32 = 0.1f0,
        difficulty_bonus_scale::Float32 = 0.1f0,
        efficiency_weight::Float32 = 0.05f0
    )
        @assert α + β + γ ≈ 1.0f0 "Reward weights must sum to 1.0"
        @assert 0.0f0 < mastery_threshold < 1.0f0 "Mastery threshold must be in (0, 1)"
        new(α, β, γ, mastery_threshold, forgetting_penalty, 
            difficulty_bonus_scale, efficiency_weight)
    end
end

"""
    RewardFunction{T<:AbstractFloat}

Complete reward computation with all components.

# Architecture
The reward function decomposes into:
1. **Gain**: Δmastery from current action
2. **Retention**: Prevention of forgetting over time
3. **Transfer**: Generalization to new topics
4. **Shaping**: Auxiliary rewards for good pedagogical practices

# Fields
- `config::RewardConfig`: Reward configuration
- `topic_similarity::Matrix{T}`: Topic-topic similarity for transfer (optional)
"""
struct RewardFunction{T<:AbstractFloat}
    config::RewardConfig
    topic_similarity::Union{Nothing, Matrix{T}}
end

"""
    RewardFunction{T}(; config=RewardConfig(), topic_similarity=nothing) where T

Construct reward function.
"""
function RewardFunction{T}(;
    config::RewardConfig = RewardConfig(),
    topic_similarity::Union{Nothing, Matrix} = nothing
) where T<:AbstractFloat
    sim = isnothing(topic_similarity) ? nothing : convert(Matrix{T}, topic_similarity)
    RewardFunction{T}(config, sim)
end

# ============================================================================
# Core Reward Components
# ============================================================================

"""
    compute_knowledge_gain(config::RewardConfig, x_prev, x_curr) -> R_gain

Compute immediate knowledge gain from action.

R_gain = Σᵢ max(masteryᵢ_curr - masteryᵢ_prev, 0)

Only positive changes count (learning). Forgetting is handled separately.
"""
function compute_knowledge_gain(config::RewardConfig, 
                                x_prev::AbstractVector,
                                x_curr::AbstractVector,
                                mastery_dim::Int)
    mastery_prev = x_prev[1:mastery_dim]
    mastery_curr = x_curr[1:mastery_dim]
    
    # Only count positive gains
    gains = max.(mastery_curr .- mastery_prev, 0.0f0)
    
    # Give extra weight to crossing the mastery threshold
    threshold_crossings = (mastery_prev .< config.mastery_threshold) .& 
                          (mastery_curr .>= config.mastery_threshold)
    
    raw_gain = sum(gains)
    threshold_bonus = sum(threshold_crossings) * 0.5f0
    
    return raw_gain + threshold_bonus
end

"""
    compute_retention(config::RewardConfig, x_prev, x_curr, time_gap) -> R_retention

Compute retention reward/penalty.

R_retention = -forgetting_penalty * time_gap * Σᵢ max(masteryᵢ_prev - threshold, 0) · decay

Penalizes allowing mastered topics to decay.
"""
function compute_retention(config::RewardConfig,
                           x_prev::AbstractVector,
                           x_curr::AbstractVector,
                           time_gap::Int,
                           mastery_dim::Int)
    mastery_prev = x_prev[1:mastery_dim]
    mastery_curr = x_curr[1:mastery_dim]
    
    # Find mastered topics that decayed
    was_mastered = mastery_prev .>= config.mastery_threshold
    decayed = max.(mastery_prev .- mastery_curr, 0.0f0)
    
    # Weight by how far above threshold they were
    above_threshold = max.(mastery_prev .- config.mastery_threshold, 0.0f0)
    
    # Penalty scales with time gap (more time = more forgetting expected)
    decay_factor = 1.0f0 - exp(-config.forgetting_penalty * Float32(time_gap))
    
    penalty = -sum(decayed .* above_threshold .* Float32.(was_mastered))
    
    return penalty * decay_factor
end

"""
    compute_transfer(config::RewardConfig, x_curr, abstraction_dim, 
                     mastery_dim, misconception_dim) -> R_transfer

Compute transfer/generalization reward.

R_transfer = mean(abstractions) - misconception_penalty

Higher abstraction levels indicate better transfer capability.
Misconceptions hinder transfer.
"""
function compute_transfer(config::RewardConfig,
                          x_curr::AbstractVector,
                          mastery_dim::Int,
                          misconception_dim::Int)
    abstraction_start = mastery_dim + misconception_dim + 1
    abstractions = x_curr[abstraction_start:end]
    misconceptions = x_curr[mastery_dim+1:mastery_dim+misconception_dim]
    
    # Positive reward for high abstraction understanding
    abstraction_score = mean(abstractions)
    
    # Penalty for active misconceptions (they block transfer)
    misconception_penalty = 0.2f0 * sum(max.(misconceptions .- 0.3f0, 0.0f0))
    
    return abstraction_score - misconception_penalty
end

# ============================================================================
# Reward Shaping (Pedagogical Best Practices)
# ============================================================================

"""
    compute_difficulty_match(x_curr, action_difficulty, mastery_dim) -> bonus

Bonus for appropriate difficulty selection.

Optimal difficulty is slightly above current mastery (zone of proximal development).
"""
function compute_difficulty_match(x_curr::AbstractVector,
                                  action_difficulty::Float32,
                                  mastery_dim::Int)
    avg_mastery = mean(x_curr[1:mastery_dim])
    
    # Ideal difficulty is ~0.1-0.2 above current mastery
    ideal_difficulty = clamp(avg_mastery + 0.15f0, 0.0f0, 1.0f0)
    
    # Gaussian-like bonus centered at ideal
    diff = abs(action_difficulty - ideal_difficulty)
    bonus = exp(-diff^2 / 0.05f0)
    
    return 0.1f0 * bonus
end

"""
    compute_efficiency_bonus(x_prev, x_curr, action, mastery_dim) -> bonus

Bonus for learning efficiency (more gain per action).

Encourages the policy to find efficient teaching strategies.
"""
function compute_efficiency_bonus(x_prev::AbstractVector,
                                  x_curr::AbstractVector,
                                  mastery_dim::Int)
    mastery_prev = x_prev[1:mastery_dim]
    mastery_curr = x_curr[1:mastery_dim]
    
    # Raw improvement
    improvement = sum(max.(mastery_curr .- mastery_prev, 0.0f0))
    
    # Efficiency is improvement normalized by current mastery level
    # (harder to improve when already good)
    current_avg = mean(mastery_curr)
    difficulty_factor = 1.0f0 + current_avg  # More bonus when improving high mastery
    
    return improvement * difficulty_factor * 0.1f0
end

# ============================================================================
# Anti-Reward-Hacking Penalties
# ============================================================================

"""
    compute_gaming_penalty(action_history, x_trajectory) -> penalty

Detect and penalize reward gaming behaviors.

Gaming behaviors:
1. Always selecting easiest problems
2. Cycling through same few problems
3. Avoiding challenging topics
"""
function compute_gaming_penalty(action_history::Vector,
                                x_trajectory::Vector;
                                window::Int = 20)
    if length(action_history) < window
        return 0.0f0
    end
    
    recent_actions = action_history[end-window+1:end]
    
    penalty = 0.0f0
    
    # Check for repetitive problem selection
    problem_ids = [a.problem_id for a in recent_actions if !isnothing(a.problem_id)]
    if !isempty(problem_ids)
        unique_problems = length(Set(problem_ids))
        repetition_ratio = 1.0f0 - Float32(unique_problems) / Float32(length(problem_ids))
        if repetition_ratio > 0.5f0
            penalty += 0.1f0 * (repetition_ratio - 0.5f0)
        end
    end
    
    # Check for consistently easy difficulty
    difficulties = [a.difficulty for a in recent_actions]
    avg_difficulty = mean(difficulties)
    if avg_difficulty < 0.3f0
        penalty += 0.05f0 * (0.3f0 - avg_difficulty)
    end
    
    return penalty
end

# ============================================================================
# Main Reward Computation
# ============================================================================

"""
    compute_reward(rf::RewardFunction, x_prev, x_curr, action; 
                   time_gap=1, config_override=nothing) -> (total, components)

Compute total reward and component breakdown.

# Arguments
- `rf::RewardFunction`: Reward function instance
- `x_prev::AbstractVector`: Previous latent state
- `x_curr::AbstractVector`: Current latent state
- `action`: Action taken
- `time_gap::Int=1`: Time since last observation
- `config_override`: Override config (for ablations)

# Returns
- `total::Real`: Total reward R(xₜ, uₜ)
- `components::NamedTuple`: Individual reward components for logging

# Mathematical Form
```
R = α·R_gain + β·R_retention + γ·R_transfer + shaping
```
"""
function compute_reward(rf::RewardFunction{T},
                        x_prev::AbstractVector,
                        x_curr::AbstractVector,
                        action;
                        time_gap::Int = 1,
                        mastery_dim::Int = 40,
                        misconception_dim::Int = 16,
                        config_override::Union{Nothing, RewardConfig} = nothing) where T
    config = isnothing(config_override) ? rf.config : config_override
    
    # Core components
    R_gain = compute_knowledge_gain(config, x_prev, x_curr, mastery_dim)
    R_retention = compute_retention(config, x_prev, x_curr, time_gap, mastery_dim)
    R_transfer = compute_transfer(config, x_curr, mastery_dim, misconception_dim)
    
    # Shaping bonuses
    difficulty_bonus = compute_difficulty_match(x_curr, action.difficulty, mastery_dim)
    efficiency_bonus = compute_efficiency_bonus(x_prev, x_curr, mastery_dim)
    
    # Combine with weights
    total = config.α * R_gain +
            config.β * R_retention +
            config.γ * R_transfer +
            config.difficulty_bonus_scale * difficulty_bonus +
            config.efficiency_weight * efficiency_bonus
    
    components = (
        gain = R_gain,
        retention = R_retention,
        transfer = R_transfer,
        difficulty_bonus = difficulty_bonus,
        efficiency_bonus = efficiency_bonus
    )
    
    return total, components
end

"""
    compute_reward_simple(rf::RewardFunction, x_prev, x_curr) -> Real

Simplified reward for quick computation (no shaping).
"""
function compute_reward_simple(rf::RewardFunction{T},
                               x_prev::AbstractVector,
                               x_curr::AbstractVector;
                               mastery_dim::Int = 40,
                               misconception_dim::Int = 16) where T
    config = rf.config
    
    R_gain = compute_knowledge_gain(config, x_prev, x_curr, mastery_dim)
    R_retention = compute_retention(config, x_prev, x_curr, 1, mastery_dim)
    R_transfer = compute_transfer(config, x_curr, mastery_dim, misconception_dim)
    
    return config.α * R_gain + config.β * R_retention + config.γ * R_transfer
end

# ============================================================================
# Cumulative Reward Computation
# ============================================================================

"""
    compute_returns(rewards::Vector, γ::Float32) -> Vector

Compute discounted returns: Gₜ = Σₖ γᵏ Rₜ₊ₖ
"""
function compute_returns(rewards::AbstractVector{T}, γ::T) where T
    n = length(rewards)
    returns = similar(rewards)
    
    running_return = zero(T)
    for t in n:-1:1
        running_return = rewards[t] + γ * running_return
        returns[t] = running_return
    end
    
    return returns
end

"""
    compute_gae(rewards, values, γ, λ) -> (advantages, returns)

Compute Generalized Advantage Estimation (GAE).

A_t = Σₖ (γλ)ᵏ δₜ₊ₖ
where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)

# Arguments
- `rewards::Vector`: Rewards [r₁, ..., rₜ]
- `values::Vector`: Value estimates [V(s₁), ..., V(sₜ)]
- `γ::Real`: Discount factor
- `λ::Real`: GAE parameter

# Returns
- `advantages::Vector`: Advantage estimates
- `returns::Vector`: Return targets for value function
"""
function compute_gae(rewards::AbstractVector{T},
                     values::AbstractVector{T},
                     γ::T,
                     λ::T) where T
    n = length(rewards)
    @assert length(values) == n
    
    advantages = similar(rewards)
    returns = similar(rewards)
    
    # Compute TD residuals
    next_value = zero(T)
    gae = zero(T)
    
    for t in n:-1:1
        # δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        if t == n
            delta = rewards[t] - values[t]  # Terminal: no next value
        else
            delta = rewards[t] + γ * values[t+1] - values[t]
        end
        
        # GAE: A_t = δₜ + γλ A_{t+1}
        gae = delta + γ * λ * gae
        advantages[t] = gae
        
        # Return target: V_target = A_t + V(s_t)
        returns[t] = advantages[t] + values[t]
    end
    
    return advantages, returns
end

# ============================================================================
# Reward Logging and Analysis
# ============================================================================

"""
    RewardLog

Accumulator for reward statistics across training.
"""
mutable struct RewardLog
    total_rewards::Vector{Float32}
    gain_rewards::Vector{Float32}
    retention_rewards::Vector{Float32}
    transfer_rewards::Vector{Float32}
    episode_count::Int
end

RewardLog() = RewardLog(Float32[], Float32[], Float32[], Float32[], 0)

function log_reward!(log::RewardLog, total::Real, components::NamedTuple)
    push!(log.total_rewards, Float32(total))
    push!(log.gain_rewards, Float32(components.gain))
    push!(log.retention_rewards, Float32(components.retention))
    push!(log.transfer_rewards, Float32(components.transfer))
end

function summarize(log::RewardLog; window::Int = 100)
    if isempty(log.total_rewards)
        return (mean_total = 0.0f0, mean_gain = 0.0f0, 
                mean_retention = 0.0f0, mean_transfer = 0.0f0)
    end
    
    recent_start = max(1, length(log.total_rewards) - window + 1)
    
    (
        mean_total = mean(log.total_rewards[recent_start:end]),
        mean_gain = mean(log.gain_rewards[recent_start:end]),
        mean_retention = mean(log.retention_rewards[recent_start:end]),
        mean_transfer = mean(log.transfer_rewards[recent_start:end])
    )
end
