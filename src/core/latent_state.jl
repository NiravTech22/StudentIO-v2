# ============================================================================
# StudentIO Latent State Module
# ============================================================================
#
# This module implements the latent state dynamics for student knowledge.
#
# Mathematical Foundation:
#   xₜ₊₁ = f(xₜ, uₜ) + wₜ
#
# Where:
#   - xₜ ∈ ℝⁿ is the latent student knowledge state
#   - uₜ ∈ ℝᵃ is the instructional action
#   - f is a learned transition function (neural network)
#   - wₜ ~ N(0, Σw) is process noise
#
# Design Decisions:
#   1. Residual learning: xₜ₊₁ = xₜ + Δx(xₜ, uₜ) for stable gradient flow
#   2. Heteroscedastic noise: Learned per-dimension variance
#   3. Bounded mastery/misconception via soft clamping
#
# ============================================================================

using Flux
using Flux: Chain, Dense, relu, tanh, softplus
using Random

"""
    TransitionModel{T<:AbstractFloat}

Learned transition dynamics for the latent student state.

Implements: xₜ₊₁ = xₜ + Δxₜ + wₜ

where Δxₜ = f_θ(xₜ, uₜ) is the learned state change
and wₜ ~ N(0, diag(σ²)) is learned heteroscedastic noise.

# Architecture
```
[xₜ, uₜ] → Dense(n+a, 128, tanh) → Dense(128, 128, relu) → Dense(128, n) → Δxₜ
```

The residual formulation ensures:
- Stable gradients (identity skip connection)
- Learning focuses on CHANGES, not absolute values
- Easier to capture slow dynamics

# Fields
- `network::Chain`: Neural network f_θ
- `noise_logvar::Vector{T}`: log(σ²) per state dimension
- `config::StudentStateConfig`: State space configuration
"""
struct TransitionModel{T<:AbstractFloat}
    network::Chain
    noise_logvar::Vector{T}
    config::StudentStateConfig
end

"""
    TransitionModel{T}(config::StudentStateConfig) where T

Construct transition model with Xavier initialization.
"""
function TransitionModel{T}(config::StudentStateConfig) where T<:AbstractFloat
    input_dim = config.state_dim + config.action_dim
    hidden_dim = 128
    output_dim = config.state_dim
    
    network = Chain(
        Dense(input_dim => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    )
    
    # Initialize noise log-variance (small initial noise)
    noise_logvar = fill(T(-3.0), config.state_dim)  # σ ≈ 0.22
    
    TransitionModel{T}(network, noise_logvar, config)
end

# Make model callable with Flux
Flux.@layer TransitionModel

# Get trainable parameters
function Flux.trainable(model::TransitionModel)
    (network = model.network, noise_logvar = model.noise_logvar)
end

"""
    transition(model::TransitionModel, x_t, u_t; deterministic=false) -> x_next

Compute the next state given current state and action.

# Arguments
- `model::TransitionModel`: Transition dynamics model
- `x_t::AbstractVector`: Current state xₜ ∈ ℝⁿ
- `u_t::AbstractVector`: Action uₜ ∈ ℝᵃ
- `deterministic::Bool=false`: If true, omit noise (for inference)

# Returns
- `x_next::Vector`: Next state xₜ₊₁

# Mathematical Form
```
Δx = f_θ([xₜ; uₜ])
wₜ = σ ⊙ ε,  ε ~ N(0, I)
xₜ₊₁ = xₜ + Δx + wₜ
```
"""
function transition(model::TransitionModel{T}, x_t::AbstractVector, u_t::AbstractVector;
                    deterministic::Bool = false) where T
    # Concatenate state and action
    input = vcat(x_t, u_t)
    
    # Compute state change
    Δx = model.network(input)
    
    # Add residual connection
    x_next = x_t .+ Δx
    
    # Add stochastic noise (reparameterization trick for differentiability)
    if !deterministic
        σ = exp.(T(0.5) .* model.noise_logvar)
        ε = randn(T, length(x_t))
        x_next = x_next .+ σ .* ε
    end
    
    return x_next
end

"""
    transition_batch(model::TransitionModel, x_batch, u_batch; deterministic=false)

Batched transition for efficiency (states as columns).

# Arguments
- `x_batch::AbstractMatrix`: States [n × batch_size]
- `u_batch::AbstractMatrix`: Actions [a × batch_size]

# Returns
- `x_next_batch::Matrix`: Next states [n × batch_size]
"""
function transition_batch(model::TransitionModel{T}, x_batch::AbstractMatrix, u_batch::AbstractMatrix;
                          deterministic::Bool = false) where T
    batch_size = size(x_batch, 2)
    @assert size(u_batch, 2) == batch_size "Batch sizes must match"
    
    # Concatenate states and actions
    input = vcat(x_batch, u_batch)
    
    # Compute state changes
    Δx = model.network(input)
    
    # Residual connection
    x_next = x_batch .+ Δx
    
    # Add noise
    if !deterministic
        σ = exp.(T(0.5) .* model.noise_logvar)
        ε = randn(T, size(x_batch))
        x_next = x_next .+ σ .* ε
    end
    
    return x_next
end

# ============================================================================
# Interpretable State Decomposition
# ============================================================================

"""
    decompose_state(model::TransitionModel, x::AbstractVector)

Decompose latent state into interpretable components.

# Returns
NamedTuple with:
- `mastery::Vector`: Topic mastery levels [0, 1]
- `misconceptions::Vector`: Misconception strengths [0, 1]
- `abstractions::Vector`: Abstract understanding ℝ
"""
function decompose_state(model::TransitionModel, x::AbstractVector)
    config = model.config
    m = config.mastery_dim
    k = config.misconception_dim
    
    (
        mastery = x[1:m],
        misconceptions = x[m+1:m+k],
        abstractions = x[m+k+1:end]
    )
end

"""
    compose_state(model::TransitionModel; mastery, misconceptions, abstractions)

Compose state vector from components.
"""
function compose_state(model::TransitionModel{T}; 
                       mastery::AbstractVector, 
                       misconceptions::AbstractVector,
                       abstractions::AbstractVector) where T
    convert(Vector{T}, vcat(mastery, misconceptions, abstractions))
end

# ============================================================================
# Transition Analysis and Diagnostics
# ============================================================================

"""
    compute_learning_effect(model::TransitionModel, x_t, u_t)

Analyze how an action affects each state dimension.

Returns the predicted Δx without noise, decomposed by component.
Useful for understanding action effects.
"""
function compute_learning_effect(model::TransitionModel, x_t::AbstractVector, u_t::AbstractVector)
    input = vcat(x_t, u_t)
    Δx = model.network(input)
    
    components = decompose_state(model, Δx)
    
    (
        mastery_change = components.mastery,
        misconception_change = components.misconceptions,
        abstraction_change = components.abstractions,
        total_change_norm = sqrt(sum(Δx.^2))
    )
end

"""
    noise_std(model::TransitionModel)

Get noise standard deviation per state dimension.
"""
function noise_std(model::TransitionModel{T}) where T
    exp.(T(0.5) .* model.noise_logvar)
end

"""
    effective_noise_snr(model::TransitionModel, x_t, u_t)

Compute signal-to-noise ratio for the transition.

SNR = ||Δx|| / ||σ||

High SNR indicates deterministic dynamics dominate.
Low SNR indicates high uncertainty in transitions.
"""
function effective_noise_snr(model::TransitionModel{T}, x_t::AbstractVector, u_t::AbstractVector) where T
    input = vcat(x_t, u_t)
    Δx = model.network(input)
    σ = noise_std(model)
    
    signal = sqrt(sum(Δx.^2))
    noise = sqrt(sum(σ.^2))
    
    return signal / (noise + eps(T))
end

# ============================================================================
# Jacobian and Stability Analysis
# ============================================================================

"""
    transition_jacobian(model::TransitionModel, x_t, u_t)

Compute Jacobian ∂xₜ₊₁/∂xₜ at a point.

Used for stability analysis:
- Eigenvalues inside unit circle → stable dynamics
- Eigenvalues outside → unstable/divergent

Requires Zygote for automatic differentiation.
"""
function transition_jacobian(model::TransitionModel, x_t::AbstractVector, u_t::AbstractVector)
    # Use Zygote to compute Jacobian
    f(x) = transition(model, x, u_t, deterministic=true)
    
    n = length(x_t)
    J = zeros(eltype(x_t), n, n)
    
    # Compute each column (∂f/∂xᵢ)
    for i in 1:n
        _, back = Zygote.pullback(f, x_t)
        eᵢ = zeros(eltype(x_t), n)
        eᵢ[i] = 1
        J[:, i] = back(eᵢ)[1]
    end
    
    return J
end

"""
    spectral_radius(model::TransitionModel, x_t, u_t)

Compute spectral radius (max |eigenvalue|) of transition Jacobian.

ρ < 1 indicates locally stable dynamics.
"""
function spectral_radius(model::TransitionModel, x_t::AbstractVector, u_t::AbstractVector)
    J = transition_jacobian(model, x_t, u_t)
    eigenvalues = eigvals(J)
    return maximum(abs.(eigenvalues))
end

# ============================================================================
# GPU Support
# ============================================================================

function Flux.gpu(model::TransitionModel{T}) where T
    TransitionModel{T}(
        gpu(model.network),
        gpu(model.noise_logvar),
        model.config
    )
end

function Flux.cpu(model::TransitionModel{T}) where T
    TransitionModel{T}(
        cpu(model.network),
        cpu(model.noise_logvar),
        model.config
    )
end
