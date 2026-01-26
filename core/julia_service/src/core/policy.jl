# ============================================================================
# StudentIO Policy Network
# ============================================================================
#
# This module implements the policy network that selects instructional actions
# based on the current belief state.
#
# Mathematical Foundation:
#   uₜ = π(hₜ)
#
# The policy optimizes:
#   J = E[Σₜ R(xₜ, uₜ)]
#
# Where:
#   - hₜ is the belief state from the belief filter
#   - uₜ is the instructional action
#   - R is the learning-centric reward
#
# Implementation: Actor-Critic with PPO for stability
#
# Action Space:
#   - Categorical: action type, problem selection
#   - Continuous: difficulty, pacing (Beta distribution for [0,1] bounds)
#
# ============================================================================

using Flux
using Flux: Chain, Dense, softmax, sigmoid, tanh, relu, softplus
using Distributions
using Random

"""
    PolicyNetwork{T<:AbstractFloat}

Actor-Critic policy network for instructional action selection.

# Architecture
```
hₜ → SharedEncoder → ─┬─→ Actor  → (action_logits, continuous_params)
                      └─→ Critic → V(hₜ)
```

# Action Distribution
- Categorical actions: Softmax over discrete choices
- Continuous actions: Beta(α, β) for bounded [0, 1] parameters

# Fields
- `shared_encoder::Chain`: Shared feature extraction
- `actor_head::Chain`: Policy output layers
- `critic_head::Chain`: Value function layers
- `action_space::ActionSpace`: Action space configuration
- `config::StudentStateConfig`: State configuration
"""
struct PolicyNetwork{T<:AbstractFloat}
    shared_encoder::Chain
    actor_head::Chain
    critic_head::Chain
    action_space::ActionSpace
    config::StudentStateConfig
end

"""
    PolicyNetwork{T}(config::StudentStateConfig; action_space=ActionSpace()) where T

Construct policy network from configuration.
"""
function PolicyNetwork{T}(config::StudentStateConfig;
                          action_space::ActionSpace = ActionSpace()) where T<:AbstractFloat
    hidden_dim = config.belief_dim
    shared_dim = 128
    
    # Shared encoder for actor and critic
    shared_encoder = Chain(
        Dense(hidden_dim => shared_dim, relu),
        Dense(shared_dim => shared_dim, relu)
    )
    
    # Actor head outputs:
    # - action_type logits: num_action_types
    # - problem logits: num_problems (can be large, so we use embedding)
    # - continuous params: 2 * continuous_dims (α, β for each Beta distribution)
    actor_output_dim = action_space.num_action_types + 
                       action_space.num_problems + 
                       2 * action_space.continuous_dims
    
    actor_head = Chain(
        Dense(shared_dim => 64, relu),
        Dense(64 => actor_output_dim)
    )
    
    # Critic head outputs scalar value estimate
    critic_head = Chain(
        Dense(shared_dim => 64, relu),
        Dense(64 => 1)
    )
    
    PolicyNetwork{T}(shared_encoder, actor_head, critic_head, action_space, config)
end

Flux.@layer PolicyNetwork

function Flux.trainable(model::PolicyNetwork)
    (
        shared_encoder = model.shared_encoder,
        actor_head = model.actor_head,
        critic_head = model.critic_head
    )
end

# ============================================================================
# Forward Pass and Action Distribution
# ============================================================================

"""
    forward(policy::PolicyNetwork, h_t) -> (shared_features, action_params, value)

Compute shared features, action parameters, and value estimate.
"""
function forward(policy::PolicyNetwork{T}, h_t::AbstractVector) where T
    # Shared encoding
    features = policy.shared_encoder(h_t)
    
    # Actor output
    actor_output = policy.actor_head(features)
    
    # Critic output
    value = policy.critic_head(features)[1]
    
    return features, actor_output, value
end

"""
    parse_action_params(policy::PolicyNetwork, actor_output)

Parse raw actor output into structured action parameters.

# Returns
NamedTuple with:
- `action_type_logits::Vector`: Logits for action type
- `problem_logits::Vector`: Logits for problem selection
- `continuous_α::Vector`: Alpha params for Beta distributions
- `continuous_β::Vector`: Beta params for Beta distributions
"""
function parse_action_params(policy::PolicyNetwork{T}, actor_output::AbstractVector) where T
    space = policy.action_space
    
    idx = 1
    
    # Action type logits
    action_type_end = idx + space.num_action_types - 1
    action_type_logits = actor_output[idx:action_type_end]
    idx = action_type_end + 1
    
    # Problem logits
    problem_end = idx + space.num_problems - 1
    problem_logits = actor_output[idx:problem_end]
    idx = problem_end + 1
    
    # Continuous parameters (α, β for each)
    n_cont = space.continuous_dims
    raw_params = actor_output[idx:end]
    
    # Use softplus to ensure positive α, β > 0
    continuous_α = softplus.(raw_params[1:n_cont]) .+ T(1.0)  # α > 1 for mode in (0,1)
    continuous_β = softplus.(raw_params[n_cont+1:end]) .+ T(1.0)
    
    (
        action_type_logits = action_type_logits,
        problem_logits = problem_logits,
        continuous_α = continuous_α,
        continuous_β = continuous_β
    )
end

# ============================================================================
# Action Selection
# ============================================================================

"""
    select_action(policy::PolicyNetwork, h_t; deterministic=false) -> (action, log_prob)

Select an action given belief state.

# Arguments
- `policy::PolicyNetwork`: The policy network
- `h_t::AbstractVector`: Belief state
- `deterministic::Bool=false`: If true, select mode of distributions

# Returns
- `action::NamedTuple`: Selected action with all components
- `log_prob::Real`: Log probability of selected action (for policy gradient)
"""
function select_action(policy::PolicyNetwork{T}, h_t::AbstractVector;
                       deterministic::Bool = false) where T
    _, actor_output, _ = forward(policy, h_t)
    params = parse_action_params(policy, actor_output)
    
    # === Action Type (Categorical) ===
    action_type_probs = softmax(params.action_type_logits)
    if deterministic
        action_type_idx = argmax(action_type_probs)
    else
        action_type_idx = sample_categorical(action_type_probs)
    end
    action_type = ActionType(action_type_idx - 1)  # 0-indexed enum
    log_prob_type = log(action_type_probs[action_type_idx] + T(1e-8))
    
    # === Problem Selection (Categorical) ===
    problem_probs = softmax(params.problem_logits)
    if deterministic
        problem_id = argmax(problem_probs)
    else
        problem_id = sample_categorical(problem_probs)
    end
    log_prob_problem = log(problem_probs[problem_id] + T(1e-8))
    
    # === Continuous Actions (Beta distributions) ===
    continuous_values = Vector{T}(undef, length(params.continuous_α))
    log_prob_continuous = zero(T)
    
    for i in eachindex(params.continuous_α)
        α, β = params.continuous_α[i], params.continuous_β[i]
        dist = Beta(α, β)
        
        if deterministic
            # Mode of Beta: (α - 1) / (α + β - 2)
            continuous_values[i] = (α - 1) / (α + β - 2)
            continuous_values[i] = clamp(continuous_values[i], T(0.01), T(0.99))
        else
            continuous_values[i] = rand(dist)
        end
        
        log_prob_continuous += logpdf(dist, continuous_values[i])
    end
    
    # Construct action
    difficulty = continuous_values[1]
    pacing = length(continuous_values) > 1 ? continuous_values[2] : T(0.5)
    emphasis = length(continuous_values) > 2 ? continuous_values[3] : T(0.5)
    
    action = (
        action_type = action_type,
        problem_id = problem_id,
        topic_id = nothing,  # Derived from problem
        difficulty = difficulty,
        pacing = pacing,
        emphasis = emphasis
    )
    
    log_prob = log_prob_type + log_prob_problem + log_prob_continuous
    
    return action, log_prob
end

"""
    sample_categorical(probs::AbstractVector) -> Int

Sample from categorical distribution given probability vector.
"""
function sample_categorical(probs::AbstractVector{T}) where T
    r = rand(T)
    cumsum = zero(T)
    for i in eachindex(probs)
        cumsum += probs[i]
        if r <= cumsum
            return i
        end
    end
    return length(probs)  # Fallback
end

# ============================================================================
# Log Probability Computation
# ============================================================================

"""
    log_prob(policy::PolicyNetwork, h_t, action) -> Real

Compute log probability of a given action under the current policy.

Used for importance sampling in PPO.
"""
function log_prob(policy::PolicyNetwork{T}, h_t::AbstractVector, action) where T
    _, actor_output, _ = forward(policy, h_t)
    params = parse_action_params(policy, actor_output)
    
    lp = zero(T)
    
    # Action type
    action_type_probs = softmax(params.action_type_logits)
    action_type_idx = Int(action.action_type) + 1
    lp += log(action_type_probs[action_type_idx] + T(1e-8))
    
    # Problem selection
    problem_probs = softmax(params.problem_logits)
    lp += log(problem_probs[action.problem_id] + T(1e-8))
    
    # Continuous actions
    continuous_vals = [action.difficulty, action.pacing, action.emphasis]
    for i in eachindex(params.continuous_α)
        if i <= length(continuous_vals)
            α, β = params.continuous_α[i], params.continuous_β[i]
            dist = Beta(α, β)
            val = clamp(continuous_vals[i], T(0.001), T(0.999))
            lp += logpdf(dist, val)
        end
    end
    
    return lp
end

# ============================================================================
# Value Function
# ============================================================================

"""
    value_estimate(policy::PolicyNetwork, h_t) -> Real

Estimate value of belief state: V(hₜ) = E[Σₜ′≥ₜ R_t′ | hₜ]
"""
function value_estimate(policy::PolicyNetwork{T}, h_t::AbstractVector) where T
    _, _, value = forward(policy, h_t)
    return value
end

"""
    value_estimate_batch(policy::PolicyNetwork, h_batch) -> Vector

Batched value estimation.
"""
function value_estimate_batch(policy::PolicyNetwork{T}, h_batch::AbstractMatrix) where T
    features = policy.shared_encoder(h_batch)
    values = policy.critic_head(features)
    return vec(values)
end

# ============================================================================
# Action Encoding (for belief filter input)
# ============================================================================

"""
    encode_action(policy::PolicyNetwork, action) -> Vector

Encode action as dense vector for belief filter.
"""
function encode_action(policy::PolicyNetwork{T}, action) where T
    space = policy.action_space
    dim = policy.config.action_dim
    
    encoded = zeros(T, dim)
    
    # One-hot for action type (first few dimensions)
    n_types = min(space.num_action_types, dim ÷ 2)
    type_idx = Int(action.action_type) + 1
    if type_idx <= n_types
        encoded[type_idx] = one(T)
    end
    
    # Continuous values
    offset = n_types
    if offset + 1 <= dim
        encoded[offset + 1] = action.difficulty
    end
    if offset + 2 <= dim
        encoded[offset + 2] = action.pacing
    end
    if offset + 3 <= dim
        encoded[offset + 3] = action.emphasis
    end
    
    # Problem embedding (normalized ID)
    if offset + 4 <= dim
        encoded[offset + 4] = T(action.problem_id) / T(space.num_problems)
    end
    
    return encoded
end

# ============================================================================
# Action Explanation (Interpretability)
# ============================================================================

"""
    explain_action(policy::PolicyNetwork, h_t, action) -> ActionRationale

Generate explanation for why an action was selected.

Provides:
- Value estimate
- Uncertainty
- Most influential belief dimensions
- Alternative actions considered
"""
function explain_action(policy::PolicyNetwork{T}, h_t::AbstractVector, action) where T
    _, actor_output, value = forward(policy, h_t)
    params = parse_action_params(policy, actor_output)
    
    # Get top alternative actions
    action_type_probs = softmax(params.action_type_logits)
    problem_probs = softmax(params.problem_logits)
    
    # Find top-k alternatives
    top_types = sortperm(action_type_probs, rev=true)[1:min(3, length(action_type_probs))]
    top_problems = sortperm(problem_probs, rev=true)[1:min(3, length(problem_probs))]
    
    alternatives = Tuple{InstructionalAction, Float32}[]
    for type_idx in top_types
        for prob_idx in top_problems
            if type_idx != Int(action.action_type) + 1 || prob_idx != action.problem_id
                alt_action = InstructionalAction(
                    ActionType(type_idx - 1),
                    prob_idx,
                    nothing,
                    action.difficulty,
                    action.pacing,
                    action.emphasis
                )
                alt_prob = action_type_probs[type_idx] * problem_probs[prob_idx]
                push!(alternatives, (alt_action, Float32(alt_prob)))
                
                if length(alternatives) >= 5
                    break
                end
            end
        end
        if length(alternatives) >= 5
            break
        end
    end
    
    # Find influential belief dimensions via gradient magnitude
    # (Approximate: use absolute value of shared encoder output)
    features = policy.shared_encoder(h_t)
    feature_importance = abs.(features)
    top_dims = sortperm(vec(feature_importance), rev=true)[1:min(5, length(features))]
    
    # Estimate uncertainty from action entropy
    type_entropy = -sum(action_type_probs .* log.(action_type_probs .+ T(1e-8)))
    
    ActionRationale(
        InstructionalAction(
            action.action_type,
            action.problem_id,
            action.topic_id,
            action.difficulty,
            action.pacing,
            action.emphasis
        ),
        Float32(value),
        Float32(type_entropy),
        top_dims,
        alternatives
    )
end

# ============================================================================
# Entropy for Exploration
# ============================================================================

"""
    action_entropy(policy::PolicyNetwork, h_t) -> Real

Compute entropy of action distribution for exploration bonus.

H(π(·|h)) = H(action_type) + H(problem) + Σᵢ H(Beta_i)
"""
function action_entropy(policy::PolicyNetwork{T}, h_t::AbstractVector) where T
    _, actor_output, _ = forward(policy, h_t)
    params = parse_action_params(policy, actor_output)
    
    # Categorical entropies
    action_type_probs = softmax(params.action_type_logits)
    H_type = -sum(action_type_probs .* log.(action_type_probs .+ T(1e-8)))
    
    problem_probs = softmax(params.problem_logits)
    H_problem = -sum(problem_probs .* log.(problem_probs .+ T(1e-8)))
    
    # Beta distribution entropy
    H_continuous = zero(T)
    for i in eachindex(params.continuous_α)
        α, β = params.continuous_α[i], params.continuous_β[i]
        dist = Beta(α, β)
        H_continuous += entropy(dist)
    end
    
    return H_type + H_problem + H_continuous
end

# ============================================================================
# GPU Support
# ============================================================================

function Flux.gpu(model::PolicyNetwork{T}) where T
    PolicyNetwork{T}(
        gpu(model.shared_encoder),
        gpu(model.actor_head),
        gpu(model.critic_head),
        model.action_space,
        model.config
    )
end

function Flux.cpu(model::PolicyNetwork{T}) where T
    PolicyNetwork{T}(
        cpu(model.shared_encoder),
        cpu(model.actor_head),
        cpu(model.critic_head),
        model.action_space,
        model.config
    )
end
