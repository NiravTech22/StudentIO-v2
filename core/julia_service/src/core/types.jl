# ============================================================================
# StudentIO Core Types and Abstractions
# ============================================================================
#
# This file defines the fundamental type system for the StudentIO meta-learning
# framework. All types are designed to preserve the hidden-state interpretation
# of student learning as a latent dynamical system.
#
# Mathematical Framework:
#   - Latent state: xₜ ∈ ℝⁿ (knowledge, misconceptions, abstractions)
#   - Transition: xₜ₊₁ = f(xₜ, uₜ) + wₜ
#   - Observation: yₜ = g(xₜ) + vₜ
#   - Belief: bₜ(x) ≈ compressed posterior over xₜ
#
# ============================================================================

"""
    StudentStateConfig

Configuration for the student state space dimensionality.

The latent state xₜ is decomposed into three interpretable components:
1. **Mastery**: Knowledge levels across topics/skills
2. **Misconceptions**: Active incorrect beliefs that interfere with learning
3. **Abstractions**: Higher-order conceptual understanding enabling transfer

# Fields
- `state_dim::Int`: Total latent state dimension (n)
- `mastery_dim::Int`: Dimensions for mastery representation
- `misconception_dim::Int`: Dimensions for misconception representation
- `abstraction_dim::Int`: Dimensions for abstraction/transfer representation
- `action_dim::Int`: Instructional action space dimension
- `observation_dim::Int`: Student observation dimension
- `belief_dim::Int`: Belief state (RNN hidden) dimension

# Mathematical Interpretation
The state vector xₜ ∈ ℝⁿ is partitioned as:
```
xₜ = [mastery₁...masteryₘ | misconcept₁...misconceptₖ | abstract₁...abstractₐ]
```
where m + k + a = n.
"""
struct StudentStateConfig
    state_dim::Int
    mastery_dim::Int
    misconception_dim::Int
    abstraction_dim::Int
    action_dim::Int
    observation_dim::Int
    belief_dim::Int
    
    function StudentStateConfig(;
        state_dim::Int = 64,
        mastery_dim::Int = 40,
        misconception_dim::Int = 16,
        abstraction_dim::Int = 8,
        action_dim::Int = 16,
        observation_dim::Int = 8,
        belief_dim::Int = 128
    )
        @assert mastery_dim + misconception_dim + abstraction_dim == state_dim (
            "State dimensions must sum to state_dim: " *
            "$mastery_dim + $misconception_dim + $abstraction_dim ≠ $state_dim"
        )
        @assert state_dim > 0 "state_dim must be positive"
        @assert belief_dim > 0 "belief_dim must be positive"
        @assert action_dim > 0 "action_dim must be positive"
        @assert observation_dim > 0 "observation_dim must be positive"
        
        new(state_dim, mastery_dim, misconception_dim, abstraction_dim,
            action_dim, observation_dim, belief_dim)
    end
end

# ============================================================================
# Student State Representation
# ============================================================================

"""
    StudentState{T<:AbstractFloat}

Represents the latent knowledge state xₜ of a student at time t.

This is the TRUE hidden state that we cannot observe directly.
The belief filter maintains an ESTIMATE of this state.

# Fields
- `mastery::Vector{T}`: Knowledge mastery levels ∈ [0, 1] per topic
- `misconceptions::Vector{T}`: Active misconception strengths ∈ [0, 1]
- `abstractions::Vector{T}`: Higher-order understanding ∈ ℝ

# Access Patterns
```julia
state = StudentState(config)
state.mastery[topic_id]          # Mastery of specific topic
state.misconceptions[misconcept_id]  # Strength of misconception
state.abstractions[concept_id]   # Abstract understanding level
to_vector(state)                 # Flatten to ℝⁿ
```
"""
struct StudentState{T<:AbstractFloat}
    mastery::Vector{T}
    misconceptions::Vector{T}
    abstractions::Vector{T}
end

"""
    StudentState{T}(config::StudentStateConfig) where T

Create a zero-initialized student state from configuration.
"""
function StudentState{T}(config::StudentStateConfig) where T<:AbstractFloat
    StudentState{T}(
        zeros(T, config.mastery_dim),
        zeros(T, config.misconception_dim),
        zeros(T, config.abstraction_dim)
    )
end

"""
    StudentState(config::StudentStateConfig)

Create Float32 student state (default precision).
"""
StudentState(config::StudentStateConfig) = StudentState{Float32}(config)

"""
    to_vector(state::StudentState) -> Vector

Flatten student state to a single vector xₜ ∈ ℝⁿ.
"""
function to_vector(state::StudentState{T}) where T
    vcat(state.mastery, state.misconceptions, state.abstractions)
end

"""
    from_vector(::Type{StudentState{T}}, v::Vector, config::StudentStateConfig) -> StudentState

Reconstruct StudentState from flattened vector.
"""
function from_vector(::Type{StudentState{T}}, v::AbstractVector, config::StudentStateConfig) where T
    @assert length(v) == config.state_dim "Vector length must match state_dim"
    
    m = config.mastery_dim
    k = config.misconception_dim
    a = config.abstraction_dim
    
    StudentState{T}(
        v[1:m],
        v[m+1:m+k],
        v[m+k+1:m+k+a]
    )
end

# ============================================================================
# Observation Types
# ============================================================================

"""
    ObservationType

Enumeration of observable student response types.

Each type has different noise characteristics and information content:
- `CORRECTNESS`: Binary correct/incorrect (high noise, common)
- `RESPONSE_TIME`: Log-normalized response latency (medium noise)
- `CONFIDENCE`: Self-reported confidence 0-1 (medium noise, subject to bias)
- `PARTIAL_CREDIT`: Continuous score 0-1 (low noise, expensive to obtain)
- `CODE_QUALITY`: Multi-dimensional code metrics (low noise, domain-specific)
- `EXPLANATION`: Natural language explanation (highest information, hardest to process)
"""
@enum ObservationType begin
    CORRECTNESS
    RESPONSE_TIME
    CONFIDENCE
    PARTIAL_CREDIT
    CODE_QUALITY
    EXPLANATION
end

"""
    Observation{T<:AbstractFloat}

A single observation yₜ from the student.

# Fields
- `obs_type::ObservationType`: What was observed
- `values::Vector{T}`: Observation values (dimension varies by type)
- `metadata::Dict{Symbol, Any}`: Additional context (problem ID, timestamp, etc.)
"""
struct Observation{T<:AbstractFloat}
    obs_type::ObservationType
    values::Vector{T}
    metadata::Dict{Symbol, Any}
end

"""
    Observation{T}(obs_type, values; metadata=Dict())

Convenience constructor for observations.
"""
function Observation{T}(obs_type::ObservationType, values::AbstractVector; 
                        metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T
    Observation{T}(obs_type, convert(Vector{T}, values), metadata)
end

# ============================================================================
# Action Space
# ============================================================================

"""
    ActionType

Types of instructional actions the system can take.

- `PRESENT_PROBLEM`: Show a new problem at specified difficulty
- `PROVIDE_HINT`: Give a hint for current problem
- `PROVIDE_SOLUTION`: Show full solution with explanation
- `REVIEW_CONCEPT`: Present conceptual review material
- `ADJUST_DIFFICULTY`: Change difficulty level for next problems
- `SWITCH_TOPIC`: Move to a different topic
- `ENCOURAGE`: Provide motivational feedback
- `PAUSE`: Suggest taking a break (fatigue detection)
"""
@enum ActionType begin
    PRESENT_PROBLEM
    PROVIDE_HINT
    PROVIDE_SOLUTION
    REVIEW_CONCEPT
    ADJUST_DIFFICULTY
    SWITCH_TOPIC
    ENCOURAGE
    PAUSE
end

"""
    ActionSpace

Defines the structure of the instructional action space.

Actions are hybrid: discrete (action type) + continuous (parameters).

# Fields
- `num_action_types::Int`: Number of discrete action types
- `num_problems::Int`: Number of problems in the problem bank
- `num_topics::Int`: Number of topics/skills
- `difficulty_range::Tuple{Float32, Float32}`: Min/max difficulty
- `continuous_dims::Int`: Total continuous action dimensions

# Action Vector Format
The policy outputs a vector [action_type_logits | problem_logits | continuous_params]:
- action_type_logits: Probability over ActionType
- problem_logits: Probability over problem bank  
- continuous_params: (difficulty, pacing, emphasis)
"""
struct ActionSpace
    num_action_types::Int
    num_problems::Int
    num_topics::Int
    difficulty_range::Tuple{Float32, Float32}
    continuous_dims::Int
    
    function ActionSpace(;
        num_problems::Int = 1000,
        num_topics::Int = 50,
        difficulty_range::Tuple{Float32, Float32} = (0.0f0, 1.0f0)
    )
        num_action_types = Int(typemax(ActionType)) - Int(typemin(ActionType)) + 1
        continuous_dims = 3  # difficulty, pacing, emphasis
        new(num_action_types, num_problems, num_topics, difficulty_range, continuous_dims)
    end
end

"""
Total dimension of action vector for policy network.
"""
function action_vector_dim(space::ActionSpace)
    space.num_action_types + space.num_problems + space.continuous_dims
end

"""
    InstructionalAction

A concrete instructional action with all parameters.

# Fields
- `action_type::ActionType`: What type of action
- `problem_id::Union{Int, Nothing}`: Which problem (if applicable)
- `topic_id::Union{Int, Nothing}`: Which topic (if applicable)
- `difficulty::Float32`: Target difficulty level
- `pacing::Float32`: Time pressure (0=relaxed, 1=timed)
- `emphasis::Float32`: How much to emphasize this (0=subtle, 1=explicit)
"""
struct InstructionalAction
    action_type::ActionType
    problem_id::Union{Int, Nothing}
    topic_id::Union{Int, Nothing}
    difficulty::Float32
    pacing::Float32
    emphasis::Float32
end

# ============================================================================
# Trajectory and Episode Types
# ============================================================================

"""
    Timestep{T<:AbstractFloat}

A single timestep in a student learning trajectory.

Contains all information for one interaction cycle.
"""
struct Timestep{T<:AbstractFloat}
    t::Int                          # Time index
    observation::Vector{T}          # yₜ encoded
    action::Vector{T}               # uₜ encoded
    reward::T                       # Rₜ
    belief_state::Vector{T}         # hₜ
    log_prob::T                     # log π(uₜ|hₜ)
    value_estimate::T               # V(hₜ)
    true_state::Union{Nothing, Vector{T}}  # xₜ (only for synthetic students)
end

"""
    Episode{T<:AbstractFloat}

A complete learning episode (sequence of timesteps).

# Fields
- `timesteps::Vector{Timestep{T}}`: Sequential interaction data
- `total_reward::T`: Cumulative reward
- `student_id::String`: Identifier for the student
- `metadata::Dict{Symbol, Any}`: Episode metadata
"""
struct Episode{T<:AbstractFloat}
    timesteps::Vector{Timestep{T}}
    total_reward::T
    student_id::String
    metadata::Dict{Symbol, Any}
end

"""
    Episode{T}()

Create empty episode.
"""
Episode{T}(student_id::String = "unknown") where T = Episode{T}(
    Timestep{T}[],
    zero(T),
    student_id,
    Dict{Symbol, Any}()
)

"""
    push!(episode::Episode, timestep::Timestep)

Add a timestep to the episode.
"""
function Base.push!(episode::Episode{T}, timestep::Timestep{T}) where T
    push!(episode.timesteps, timestep)
end

"""
    length(episode::Episode)

Number of timesteps in episode.
"""
Base.length(episode::Episode) = length(episode.timesteps)

# ============================================================================
# Diagnostic Types
# ============================================================================

"""
    BeliefDiagnostics{T<:AbstractFloat}

Diagnostics for monitoring belief filter quality.

# Fields
- `belief_mse::T`: Mean squared error vs true state (synthetic only)
- `belief_drift::Vector{T}`: ||hₜ - hₜ₋₁|| over time
- `uncertainty_trajectory::Vector{T}`: Uncertainty estimates over time
- `calibration_error::T`: Difference between predicted and actual error
- `collapse_detected::Bool`: Whether uncertainty collapsed inappropriately
"""
struct BeliefDiagnostics{T<:AbstractFloat}
    belief_mse::T
    belief_drift::Vector{T}
    uncertainty_trajectory::Vector{T}
    calibration_error::T
    collapse_detected::Bool
end

"""
    ActionRationale

Explanation for why an action was selected.

Used for interpretability and teacher oversight.

# Fields
- `action::InstructionalAction`: The selected action
- `value_estimate::Float32`: Expected future reward
- `uncertainty::Float32`: Belief uncertainty at decision time
- `top_belief_dims::Vector{Int}`: Most influential belief dimensions
- `alternative_actions::Vector{Tuple{InstructionalAction, Float32}}`: Runner-up actions with values
"""
struct ActionRationale
    action::InstructionalAction
    value_estimate::Float32
    uncertainty::Float32
    top_belief_dims::Vector{Int}
    alternative_actions::Vector{Tuple{InstructionalAction, Float32}}
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    clamp_state!(state::StudentState, lo=0, hi=1)

Clamp mastery and misconception values to valid range [lo, hi].
Abstractions are left unclamped (can be negative for anti-patterns).
"""
function clamp_state!(state::StudentState{T}, lo::T = zero(T), hi::T = one(T)) where T
    clamp!(state.mastery, lo, hi)
    clamp!(state.misconceptions, lo, hi)
    return state
end

"""
    state_norm(state::StudentState)

Compute L2 norm of state vector (for diagnostics).
"""
function state_norm(state::StudentState)
    sqrt(sum(state.mastery.^2) + sum(state.misconceptions.^2) + sum(state.abstractions.^2))
end

"""
    mastery_mean(state::StudentState)

Average mastery level across all topics.
"""
mastery_mean(state::StudentState) = mean(state.mastery)

"""
    has_misconceptions(state::StudentState; threshold=0.3)

Check if any misconception exceeds threshold.
"""
has_misconceptions(state::StudentState; threshold=0.3f0) = any(state.misconceptions .> threshold)
