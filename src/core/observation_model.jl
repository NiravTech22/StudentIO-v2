# ============================================================================
# StudentIO Observation Model
# ============================================================================
#
# This module implements the observation model that maps latent student state
# to observable evidence.
#
# Mathematical Foundation:
#   yₜ = g(xₜ) + vₜ
#
# Where:
#   - xₜ ∈ ℝⁿ is the latent student knowledge state
#   - yₜ ∈ ℝᵐ is the observation (response, timing, confidence, etc.)
#   - g is a learned observation function (neural network)
#   - vₜ ~ N(0, Σᵥ) is measurement noise
#
# Observation Types:
#   1. CORRECTNESS: Binary (Bernoulli likelihood)
#   2. RESPONSE_TIME: Log-normal (positive, right-skewed)
#   3. CONFIDENCE: Beta (bounded [0,1])
#   4. PARTIAL_CREDIT: Truncated Gaussian [0,1]
#
# ============================================================================

using Flux
using Flux: Chain, Dense, sigmoid, softplus, tanh
using Distributions

"""
    ObservationModel{T<:AbstractFloat}

Maps latent student state to observable outputs with learned noise model.

Implements: yₜ = g_θ(xₜ) + vₜ

with heteroscedastic noise where variance may depend on state.

# Architecture
```
xₜ → Dense(n, 64, tanh) → Dense(64, 64, relu) → Dense(64, 2m) → [μ, log σ²]
```

The output dimension is 2m: m for observation means, m for log-variances.

# Fields
- `encoder::Chain`: Neural network g_θ mapping state to observation params
- `observation_dim::Int`: Dimension of observation space
- `config::StudentStateConfig`: State configuration
"""
struct ObservationModel{T<:AbstractFloat}
    encoder::Chain
    observation_dim::Int
    config::StudentStateConfig
end

"""
    ObservationModel{T}(config::StudentStateConfig) where T

Construct observation model from configuration.
"""
function ObservationModel{T}(config::StudentStateConfig) where T<:AbstractFloat
    input_dim = config.state_dim
    hidden_dim = 64
    # Output: [correctness_logit, response_time_params, confidence_params, partial_credit_params]
    # Each observation type gets mean + log_var = 2 params (except correctness = 1 logit)
    output_dim = config.observation_dim * 2  # mean + log_var per dimension
    
    encoder = Chain(
        Dense(input_dim => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    )
    
    ObservationModel{T}(encoder, config.observation_dim, config)
end

Flux.@layer ObservationModel

function Flux.trainable(model::ObservationModel)
    (encoder = model.encoder,)
end

"""
    observe(model::ObservationModel, x_t) -> (μ, σ)

Compute observation distribution parameters given latent state.

# Arguments
- `model::ObservationModel`: The observation model
- `x_t::AbstractVector`: Latent state xₜ

# Returns
- `(μ, σ)`: Mean and standard deviation for each observation dimension

# Note
The first dimension is treated specially as correctness (Bernoulli).
Remaining dimensions are Gaussian with learned heteroscedastic variance.
"""
function observe(model::ObservationModel{T}, x_t::AbstractVector) where T
    output = model.encoder(x_t)
    
    m = model.observation_dim
    
    # Split into means and log-variances
    μ = output[1:m]
    log_var = output[m+1:2m]
    
    # Compute standard deviation with softplus for numerical stability
    σ = sqrt.(softplus.(log_var) .+ T(1e-6))
    
    return μ, σ
end

"""
    observe_batch(model::ObservationModel, x_batch) -> (μ_batch, σ_batch)

Batched observation for efficiency.

# Arguments
- `x_batch::AbstractMatrix`: States [n × batch_size]

# Returns
- `(μ_batch, σ_batch)`: Parameters [m × batch_size] each
"""
function observe_batch(model::ObservationModel{T}, x_batch::AbstractMatrix) where T
    output = model.encoder(x_batch)
    
    m = model.observation_dim
    
    μ = output[1:m, :]
    log_var = output[m+1:2m, :]
    σ = sqrt.(softplus.(log_var) .+ T(1e-6))
    
    return μ, σ
end

"""
    sample_observation(model::ObservationModel, x_t) -> y_t

Sample an observation given latent state.

Uses reparameterization trick for differentiability.
"""
function sample_observation(model::ObservationModel{T}, x_t::AbstractVector) where T
    μ, σ = observe(model, x_t)
    
    y = similar(μ)
    
    # Dimension 1: Correctness (Bernoulli)
    p_correct = sigmoid(μ[1])
    y[1] = rand() < p_correct ? one(T) : zero(T)
    
    # Remaining dimensions: Gaussian
    for i in 2:length(μ)
        y[i] = μ[i] + σ[i] * randn(T)
    end
    
    return y
end

# ============================================================================
# Likelihood Computation (for Bayesian update)
# ============================================================================

"""
    log_likelihood(model::ObservationModel, x_t, y_observed) -> log p(y|x)

Compute log-likelihood of observation given state.

This is used in the belief filter's Bayesian update.

# Likelihood Model
- Correctness (dim 1): Bernoulli with logit from encoder
- Continuous (dims 2:m): Gaussian with learned mean and variance

# Arguments
- `model::ObservationModel`: The observation model
- `x_t::AbstractVector`: Latent state
- `y_observed::AbstractVector`: Observed values

# Returns
- `ll::Real`: Log-likelihood log p(yₜ | xₜ)
"""
function log_likelihood(model::ObservationModel{T}, x_t::AbstractVector, 
                        y_observed::AbstractVector) where T
    μ, σ = observe(model, x_t)
    
    ll = zero(T)
    
    # Correctness: Bernoulli log-likelihood
    # y[1] ∈ {0, 1}, μ[1] is logit
    p_correct = sigmoid(μ[1])
    if y_observed[1] > 0.5  # Correct
        ll += log(p_correct + T(1e-8))
    else  # Incorrect
        ll += log(1 - p_correct + T(1e-8))
    end
    
    # Continuous observations: Gaussian log-likelihood
    for i in 2:length(μ)
        # log N(y | μ, σ²) = -0.5 * [(y-μ)²/σ² + log(2π) + log(σ²)]
        z = (y_observed[i] - μ[i]) / σ[i]
        ll += -T(0.5) * (z^2 + log(T(2π)) + 2*log(σ[i]))
    end
    
    return ll
end

"""
    log_likelihood_batch(model::ObservationModel, x_batch, y_batch) -> Vector

Compute log-likelihood for a batch of state-observation pairs.
"""
function log_likelihood_batch(model::ObservationModel{T}, x_batch::AbstractMatrix,
                              y_batch::AbstractMatrix) where T
    batch_size = size(x_batch, 2)
    @assert size(y_batch, 2) == batch_size
    
    μ_batch, σ_batch = observe_batch(model, x_batch)
    
    ll = zeros(T, batch_size)
    
    for j in 1:batch_size
        # Correctness
        p_correct = sigmoid(μ_batch[1, j])
        if y_batch[1, j] > 0.5
            ll[j] += log(p_correct + T(1e-8))
        else
            ll[j] += log(1 - p_correct + T(1e-8))
        end
        
        # Continuous
        for i in 2:size(μ_batch, 1)
            z = (y_batch[i, j] - μ_batch[i, j]) / σ_batch[i, j]
            ll[j] += -T(0.5) * (z^2 + log(T(2π)) + 2*log(σ_batch[i, j]))
        end
    end
    
    return ll
end

# ============================================================================
# Observation Encoding (for belief filter input)
# ============================================================================

"""
    encode_observation(model::ObservationModel, observation::Observation)

Encode a structured observation into a fixed-size vector for the belief filter.

Handles different observation types with appropriate normalization.
"""
function encode_observation(model::ObservationModel{T}, observation::Observation) where T
    m = model.observation_dim
    encoded = zeros(T, m)
    
    # Fill based on observation type
    if observation.obs_type == CORRECTNESS
        encoded[1] = observation.values[1]  # 0 or 1
        
    elseif observation.obs_type == RESPONSE_TIME
        # Log-normalize response time (assuming values in seconds)
        encoded[2] = log(observation.values[1] + T(0.1))
        
    elseif observation.obs_type == CONFIDENCE
        encoded[3] = observation.values[1]  # Already [0, 1]
        
    elseif observation.obs_type == PARTIAL_CREDIT
        encoded[4] = observation.values[1]  # Already [0, 1]
        
    elseif observation.obs_type == CODE_QUALITY
        # Multiple quality metrics
        n_metrics = min(length(observation.values), m - 4)
        for i in 1:n_metrics
            encoded[4 + i] = observation.values[i]
        end
    end
    
    return encoded
end

"""
    encode_observation(model::ObservationModel, obs_vector::AbstractVector)

Identity encoding when observation is already a vector.
"""
function encode_observation(model::ObservationModel{T}, obs_vector::AbstractVector) where T
    convert(Vector{T}, obs_vector)
end

# ============================================================================
# Observation Prediction and Diagnostics
# ============================================================================

"""
    predict_correctness(model::ObservationModel, x_t) -> probability

Get probability of correct response given state.
"""
function predict_correctness(model::ObservationModel, x_t::AbstractVector)
    μ, _ = observe(model, x_t)
    return sigmoid(μ[1])
end

"""
    observation_uncertainty(model::ObservationModel, x_t)

Get uncertainty estimates for each observation dimension.

High uncertainty suggests the state estimate is unreliable
for predicting that observation.
"""
function observation_uncertainty(model::ObservationModel, x_t::AbstractVector)
    _, σ = observe(model, x_t)
    return σ
end

"""
    expected_information_gain(model::ObservationModel, x_t, action)

Estimate expected information gain from taking an action.

Uses entropy reduction as proxy for information gain.
Higher values suggest the observation will be more informative.

This is used for exploration/exploitation tradeoff in action selection.
"""
function expected_information_gain(model::ObservationModel{T}, x_t::AbstractVector) where T
    _, σ = observe(model, x_t)
    
    # Approximate entropy of observation distribution
    # For Gaussian: H = 0.5 * log(2πe σ²)
    continuous_entropy = T(0.5) * sum(log.(T(2π * exp(1)) .* σ[2:end].^2))
    
    # For Bernoulli: H = -p log p - (1-p) log(1-p)
    p = sigmoid(observe(model, x_t)[1][1])
    binary_entropy = -p * log(p + T(1e-8)) - (1-p) * log(1-p + T(1e-8))
    
    return binary_entropy + continuous_entropy
end

# ============================================================================
# GPU Support
# ============================================================================

function Flux.gpu(model::ObservationModel{T}) where T
    ObservationModel{T}(
        gpu(model.encoder),
        model.observation_dim,
        model.config
    )
end

function Flux.cpu(model::ObservationModel{T}) where T
    ObservationModel{T}(
        cpu(model.encoder),
        model.observation_dim,
        model.config
    )
end
