"""
    StudentIO

A meta-learning system for adaptive education that models student learning as a 
latent dynamical system with sequential belief inference and control.

# Mathematical Foundation

The system treats each student as a latent dynamical process:
- Latent state: xₜ ∈ ℝⁿ (knowledge, misconceptions, abstractions)
- Transition: xₜ₊₁ = f(xₜ, uₜ) + wₜ
- Observation: yₜ = g(xₜ) + vₜ

The belief state hₜ approximates the posterior over xₜ given observations.
Actions are selected via policy π(hₜ) to maximize long-term learning.

# Architecture Overview

```
Student Interaction → Observation Encoder → Recurrent Belief State
                                                    ↓
                     Instructional Action ← Policy Network
```

# Modules
- `Core`: Latent state, observation model, belief filter, policy, reward
- `Meta`: Meta-learning across students, task distribution
- `Training`: Synthetic student simulation, training loop
- `Evaluation`: Diagnostics, ablation studies
- `CUDA`: GPU-accelerated operations

# Example

```julia
using StudentIO

# Create model with default configuration
model = StudentIOModel()

# Train on synthetic students
train!(model, num_episodes=10000)

# Deploy for new student
session = create_session(model)
for interaction in student_interactions
    action = step!(session, interaction)
    present_to_student(action)
end
```
"""
module StudentIO

using Flux
using Flux: Chain, Dense, GRU, softmax, sigmoid, relu
using CUDA
using Zygote
using Distributions
using LinearAlgebra
using Random
using Statistics

# Core types and abstractions
include("core/types.jl")

# Core components
include("core/latent_state.jl")
include("core/observation_model.jl")
include("core/belief_filter.jl")
include("core/policy.jl")
include("core/reward.jl")

# Meta-learning framework
include("meta/task_distribution.jl")
include("meta/meta_learning.jl")

# Training infrastructure
include("training/simulate_student.jl")
include("training/train_loop.jl")

# Evaluation and diagnostics
include("evaluation/diagnostics.jl")
include("evaluation/ablations.jl")

# CUDA acceleration
include("../cuda/kernels.jl")
include("../cuda/acceleration.jl")

# ============================================================================
# Public API Exports
# ============================================================================

# Core types
export StudentStateConfig, StudentState, ActionSpace, Observation
export TransitionModel, ObservationModel, BeliefFilter, PolicyNetwork
export RewardConfig, RewardFunction

# Model construction
export StudentIOModel, create_default_model

# Training
export SyntheticStudent, StudentTask, TaskDistribution
export sample_student, train!, meta_train!

# Inference and control
export create_session, step!, reset!, get_belief_state
export select_action, explain_action

# Evaluation
export DiagnosticResults, run_diagnostics, run_ablation

# Configuration
export PPOConfig, MetaLearnerConfig

# ============================================================================
# Main Model Structure
# ============================================================================

"""
    StudentIOModel

Complete meta-learning model for adaptive student instruction.

# Fields
- `config::StudentStateConfig`: State space configuration
- `transition::TransitionModel`: Latent state transition dynamics
- `observation::ObservationModel`: Observation mapping and likelihood
- `filter::BeliefFilter`: GRU-based belief inference
- `policy::PolicyNetwork`: Action selection policy (actor-critic)
- `reward::RewardFunction`: Learning-centric reward computation
"""
struct StudentIOModel{T<:AbstractFloat}
    config::StudentStateConfig
    transition::TransitionModel{T}
    observation::ObservationModel{T}
    filter::BeliefFilter{T}
    policy::PolicyNetwork{T}
    reward::RewardFunction{T}
end

# Move model to GPU
function Flux.gpu(model::StudentIOModel)
    StudentIOModel(
        model.config,
        gpu(model.transition),
        gpu(model.observation),
        gpu(model.filter),
        gpu(model.policy),
        model.reward  # Reward is CPU-only (simple computation)
    )
end

# Move model to CPU
function Flux.cpu(model::StudentIOModel)
    StudentIOModel(
        model.config,
        cpu(model.transition),
        cpu(model.observation),
        cpu(model.filter),
        cpu(model.policy),
        model.reward
    )
end

# Get all trainable parameters
function Flux.trainable(model::StudentIOModel)
    (
        transition = Flux.trainable(model.transition),
        observation = Flux.trainable(model.observation),
        filter = Flux.trainable(model.filter),
        policy = Flux.trainable(model.policy)
    )
end

"""
    create_default_model(; kwargs...) -> StudentIOModel

Create a StudentIOModel with sensible defaults.

# Keyword Arguments
- `state_dim::Int=64`: Total latent state dimension
- `mastery_dim::Int=40`: Knowledge mastery dimensions
- `misconception_dim::Int=16`: Active misconception dimensions
- `abstraction_dim::Int=8`: Higher-order conceptual dimensions
- `belief_dim::Int=128`: Belief state (RNN hidden) dimension
- `action_dim::Int=16`: Action space dimension
- `observation_dim::Int=8`: Observation space dimension
- `T::Type=Float32`: Numeric precision

# Returns
- `StudentIOModel{T}`: Fully initialized model
"""
function create_default_model(;
    state_dim::Int = 64,
    mastery_dim::Int = 40,
    misconception_dim::Int = 16,
    abstraction_dim::Int = 8,
    belief_dim::Int = 128,
    action_dim::Int = 16,
    observation_dim::Int = 8,
    T::Type = Float32
)
    @assert mastery_dim + misconception_dim + abstraction_dim == state_dim "State dimensions must sum to state_dim"
    
    config = StudentStateConfig(
        state_dim = state_dim,
        mastery_dim = mastery_dim,
        misconception_dim = misconception_dim,
        abstraction_dim = abstraction_dim,
        action_dim = action_dim,
        observation_dim = observation_dim,
        belief_dim = belief_dim
    )
    
    transition = TransitionModel{T}(config)
    observation = ObservationModel{T}(config)
    filter = BeliefFilter{T}(config)
    policy = PolicyNetwork{T}(config)
    reward = RewardFunction{T}()
    
    return StudentIOModel{T}(config, transition, observation, filter, policy, reward)
end

# ============================================================================
# Session Management (Inference Time)
# ============================================================================

"""
    StudentSession

Represents an active learning session with a single student.
Maintains belief state across interactions.
"""
mutable struct StudentSession{T<:AbstractFloat}
    model::StudentIOModel{T}
    belief_state::Vector{T}
    uncertainty::T
    last_action::Union{Nothing, NamedTuple}
    step_count::Int
    history::Vector{NamedTuple}
end

"""
    create_session(model::StudentIOModel) -> StudentSession

Initialize a new learning session with a student.
Belief state is initialized to the learned prior.
"""
function create_session(model::StudentIOModel{T}) where T
    belief_state = zeros(T, model.config.belief_dim)
    StudentSession{T}(
        model,
        belief_state,
        one(T),  # High initial uncertainty
        nothing,
        0,
        NamedTuple[]
    )
end

"""
    step!(session::StudentSession, observation) -> action

Process a new observation and return the next instructional action.

# Arguments
- `session::StudentSession`: Active learning session
- `observation`: Student response/observation data

# Returns
- `action::NamedTuple`: Recommended instructional action with rationale
"""
function step!(session::StudentSession, observation)
    model = session.model
    
    # Encode observation
    obs_vec = encode_observation(model.observation, observation)
    
    # Encode last action (if any)
    action_vec = if isnothing(session.last_action)
        zeros(eltype(session.belief_state), model.config.action_dim)
    else
        encode_action(model.policy, session.last_action)
    end
    
    # Update belief state
    session.belief_state, session.uncertainty = update_belief(
        model.filter,
        session.belief_state,
        obs_vec,
        action_vec
    )
    
    # Select action
    action, log_prob = select_action(model.policy, session.belief_state)
    
    # Generate rationale for interpretability
    rationale = explain_action(model.policy, session.belief_state, action)
    
    # Update session state
    session.last_action = action
    session.step_count += 1
    push!(session.history, (
        step = session.step_count,
        observation = observation,
        action = action,
        uncertainty = session.uncertainty,
        rationale = rationale
    ))
    
    return action
end

"""
    reset!(session::StudentSession)

Reset the session to initial state (new episode with same student).
"""
function reset!(session::StudentSession{T}) where T
    session.belief_state = zeros(T, session.model.config.belief_dim)
    session.uncertainty = one(T)
    session.last_action = nothing
    session.step_count = 0
    empty!(session.history)
    return session
end

"""
    get_belief_state(session::StudentSession) -> (belief, uncertainty)

Retrieve current belief state and uncertainty estimate.
"""
function get_belief_state(session::StudentSession)
    return (belief = copy(session.belief_state), uncertainty = session.uncertainty)
end

end # module StudentIO
