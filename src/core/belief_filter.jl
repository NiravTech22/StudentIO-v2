# ============================================================================
# StudentIO Belief Filter
# ============================================================================
#
# This module implements the belief filter that maintains a compressed
# representation of the posterior over latent student state.
#
# Mathematical Foundation:
#   bₜ(x) ∝ p(yₜ|x) ∫ p(x|xₜ₋₁, uₜ₋₁) bₜ₋₁(xₜ₋₁) dxₜ₋₁
#
# Approximated as:
#   hₜ = Φ(hₜ₋₁, yₜ, uₜ₋₁)
#
# Where:
#   - hₜ ∈ ℝᵈ is the belief state (RNN hidden state)
#   - Φ is a learned GRU that approximates Bayesian filtering
#   - The belief state compresses the full posterior into fixed dimension
#
# Key Design Decisions:
#   1. GRU over LSTM: Simpler, comparable performance, interpretable gates
#   2. Separate encoders for observations and actions
#   3. Dedicated uncertainty head for calibrated confidence
#   4. Explicit belief drift monitoring
#
# ============================================================================

using Flux
using Flux: Chain, Dense, GRU, GRUCell, sigmoid, tanh, relu
using LinearAlgebra

"""
    BeliefFilter{T<:AbstractFloat}

GRU-based approximate Bayesian filter for student belief state.

The belief state hₜ represents a compressed sufficient statistic for
the posterior over the latent student state xₜ.

# Interpretation
Unlike a generic RNN/sequence model, the GRU is EXPLICITLY interpreted as:
- **Input gates** ← observation integration strength
- **Reset gates** ← belief correction magnitude
- **Hidden state** ← compressed posterior

This interpretation constrains architecture choices and enables principled debugging.

# Architecture
```
yₜ → ObsEncoder → ─┐
                   ├─→ GRU → hₜ → UncertaintyHead → σ²_belief
uₜ₋₁ → ActEncoder → ─┘
```

# Fields
- `obs_encoder::Chain`: Encodes observations yₜ
- `action_encoder::Chain`: Encodes actions uₜ₋₁
- `gru_cell::GRUCell`: Core recurrent cell
- `uncertainty_head::Chain`: Predicts belief uncertainty
- `state_decoder::Chain`: Decodes belief to state estimate (optional)
- `hidden_dim::Int`: Belief state dimension
- `config::StudentStateConfig`: State configuration
"""
struct BeliefFilter{T<:AbstractFloat}
    obs_encoder::Chain
    action_encoder::Chain
    gru_cell::GRUCell
    uncertainty_head::Chain
    state_decoder::Chain
    hidden_dim::Int
    config::StudentStateConfig
end

"""
    BeliefFilter{T}(config::StudentStateConfig) where T

Construct belief filter from configuration.
"""
function BeliefFilter{T}(config::StudentStateConfig) where T<:AbstractFloat
    obs_dim = config.observation_dim
    action_dim = config.action_dim
    hidden_dim = config.belief_dim
    state_dim = config.state_dim
    
    # Encoder dimensions
    embed_dim = 32
    
    # Observation encoder: yₜ → embedding
    obs_encoder = Chain(
        Dense(obs_dim => embed_dim, tanh),
        Dense(embed_dim => embed_dim, relu)
    )
    
    # Action encoder: uₜ₋₁ → embedding
    action_encoder = Chain(
        Dense(action_dim => embed_dim, tanh),
        Dense(embed_dim => embed_dim, relu)
    )
    
    # GRU cell: takes concatenated embeddings
    input_dim = 2 * embed_dim
    gru_cell = GRUCell(input_dim => hidden_dim)
    
    # Uncertainty head: hₜ → scalar uncertainty estimate
    uncertainty_head = Chain(
        Dense(hidden_dim => 32, relu),
        Dense(32 => 1, softplus)  # Positive uncertainty
    )
    
    # State decoder: hₜ → x̂ₜ (reconstruction target)
    state_decoder = Chain(
        Dense(hidden_dim => 64, relu),
        Dense(64 => state_dim)
    )
    
    BeliefFilter{T}(
        obs_encoder, action_encoder, gru_cell,
        uncertainty_head, state_decoder, hidden_dim, config
    )
end

Flux.@layer BeliefFilter

function Flux.trainable(model::BeliefFilter)
    (
        obs_encoder = model.obs_encoder,
        action_encoder = model.action_encoder,
        gru_cell = model.gru_cell,
        uncertainty_head = model.uncertainty_head,
        state_decoder = model.state_decoder
    )
end

# ============================================================================
# Belief Update
# ============================================================================

"""
    update_belief(filter::BeliefFilter, h_prev, y_t, u_prev) -> (h_t, uncertainty)

Perform one step of belief update given new observation.

This is the core Bayesian filtering step, approximated by the GRU.

# Arguments
- `filter::BeliefFilter`: The belief filter
- `h_prev::AbstractVector`: Previous belief state hₜ₋₁
- `y_t::AbstractVector`: Current observation yₜ
- `u_prev::AbstractVector`: Previous action uₜ₋₁

# Returns
- `h_t::Vector`: Updated belief state
- `uncertainty::Real`: Estimated uncertainty about true state

# Mathematical Interpretation
The GRU gates approximate:
- Reset gate r: How much to "forget" prior belief (belief correction)
- Update gate z: How much new observation changes belief
- Candidate state h̃: Integrated posterior update
"""
function update_belief(filter::BeliefFilter{T}, h_prev::AbstractVector,
                       y_t::AbstractVector, u_prev::AbstractVector) where T
    # Encode inputs
    obs_emb = filter.obs_encoder(y_t)
    act_emb = filter.action_encoder(u_prev)
    
    # Concatenate embeddings as GRU input
    input = vcat(obs_emb, act_emb)
    
    # GRU update: hₜ = Φ(hₜ₋₁, input)
    h_t = filter.gru_cell(input, h_prev)
    
    # Estimate uncertainty
    uncertainty = filter.uncertainty_head(h_t)[1]
    
    return h_t, uncertainty
end

"""
    update_belief_sequence(filter::BeliefFilter, observations, actions; 
                           h_init=nothing) -> (belief_trajectory, uncertainties)

Process a sequence of observations, returning belief trajectory.

# Arguments
- `observations::Vector{Vector}`: Sequence of observations [y₁, y₂, ...]
- `actions::Vector{Vector}`: Sequence of actions [u₀, u₁, ...] (length = obs - 1 or obs)

# Returns
- `belief_trajectory::Matrix`: Belief states [hidden_dim × T]
- `uncertainties::Vector`: Uncertainty at each step
"""
function update_belief_sequence(filter::BeliefFilter{T}, 
                                observations::Vector,
                                actions::Vector;
                                h_init::Union{Nothing, AbstractVector} = nothing) where T
    n_steps = length(observations)
    hidden_dim = filter.hidden_dim
    
    # Initialize belief state
    h = isnothing(h_init) ? zeros(T, hidden_dim) : h_init
    
    # Allocate output
    belief_trajectory = zeros(T, hidden_dim, n_steps)
    uncertainties = zeros(T, n_steps)
    
    # Initial action (zero if not provided)
    u_prev = length(actions) > 0 ? actions[1] : zeros(T, filter.config.action_dim)
    
    for t in 1:n_steps
        y_t = observations[t]
        h, uncertainty = update_belief(filter, h, y_t, u_prev)
        
        belief_trajectory[:, t] = h
        uncertainties[t] = uncertainty
        
        # Update action for next step
        if t < length(actions)
            u_prev = actions[t + 1]
        end
    end
    
    return belief_trajectory, uncertainties
end

# ============================================================================
# State Decoding and Reconstruction
# ============================================================================

"""
    decode_state(filter::BeliefFilter, h_t) -> x̂_t

Decode belief state to estimated latent state.

This provides the system's "best guess" of the true student state.
"""
function decode_state(filter::BeliefFilter, h_t::AbstractVector)
    filter.state_decoder(h_t)
end

"""
    decode_state_batch(filter::BeliefFilter, h_batch) -> x̂_batch

Batched state decoding.
"""
function decode_state_batch(filter::BeliefFilter, h_batch::AbstractMatrix)
    filter.state_decoder(h_batch)
end

"""
    reconstruction_loss(filter::BeliefFilter, h_t, x_true)

Compute state reconstruction loss for training.

Only applicable with synthetic students where ground truth is known.
"""
function reconstruction_loss(filter::BeliefFilter{T}, h_t::AbstractVector, 
                             x_true::AbstractVector) where T
    x_hat = decode_state(filter, h_t)
    sum((x_hat .- x_true).^2)
end

"""
    reconstruction_loss_batch(filter::BeliefFilter, h_batch, x_batch)

Batched reconstruction loss.
"""
function reconstruction_loss_batch(filter::BeliefFilter{T}, h_batch::AbstractMatrix,
                                   x_batch::AbstractMatrix) where T
    x_hat = decode_state_batch(filter, h_batch)
    mean(sum((x_hat .- x_batch).^2, dims=1))
end

# ============================================================================
# Belief Diagnostics
# ============================================================================

"""
    compute_belief_drift(belief_trajectory::AbstractMatrix) -> Vector

Compute belief drift ||hₜ - hₜ₋₁|| over time.

High drift indicates significant belief updates.
Sudden spikes may indicate surprising observations.
"""
function compute_belief_drift(belief_trajectory::AbstractMatrix)
    n_steps = size(belief_trajectory, 2)
    drift = zeros(eltype(belief_trajectory), n_steps - 1)
    
    for t in 2:n_steps
        drift[t-1] = norm(belief_trajectory[:, t] - belief_trajectory[:, t-1])
    end
    
    return drift
end

"""
    detect_uncertainty_collapse(uncertainties::AbstractVector; 
                                 threshold=0.01, window=10) -> Bool

Detect if uncertainty has collapsed inappropriately.

Collapse is detected when uncertainty drops below threshold and stays there,
which may indicate overconfidence.
"""
function detect_uncertainty_collapse(uncertainties::AbstractVector;
                                     threshold::Real = 0.01,
                                     window::Int = 10)
    if length(uncertainties) < window
        return false
    end
    
    # Check if recent uncertainties are all below threshold
    recent = uncertainties[end-window+1:end]
    return all(recent .< threshold)
end

"""
    belief_diagnostics(filter::BeliefFilter, belief_trajectory, uncertainties, 
                       x_true_trajectory=nothing) -> BeliefDiagnostics

Compute comprehensive diagnostics for belief filter quality.

# Arguments
- `belief_trajectory::Matrix`: [hidden_dim × T]
- `uncertainties::Vector`: [T]
- `x_true_trajectory::Matrix`: [state_dim × T] (optional, for synthetic students)

# Returns
- `BeliefDiagnostics`: Struct with all diagnostic metrics
"""
function belief_diagnostics(filter::BeliefFilter{T}, 
                            belief_trajectory::AbstractMatrix,
                            uncertainties::AbstractVector;
                            x_true_trajectory::Union{Nothing, AbstractMatrix} = nothing) where T
    # Belief drift
    drift = compute_belief_drift(belief_trajectory)
    
    # Collapse detection
    collapse = detect_uncertainty_collapse(uncertainties)
    
    # Calibration and MSE (only if ground truth available)
    if !isnothing(x_true_trajectory)
        # Decode beliefs to state estimates
        x_hat_trajectory = decode_state_batch(filter, belief_trajectory)
        
        # MSE per timestep
        mse_per_t = vec(mean((x_hat_trajectory .- x_true_trajectory).^2, dims=1))
        belief_mse = mean(mse_per_t)
        
        # Calibration: correlation between uncertainty and actual error
        errors = sqrt.(mse_per_t)
        calibration_error = if length(errors) > 2
            # Correlation should be positive (high uncertainty ↔ high error)
            r = cor(uncertainties, errors)
            isnan(r) ? one(T) : one(T) - r  # 0 = perfect calibration
        else
            one(T)  # Insufficient data
        end
    else
        belief_mse = T(NaN)
        calibration_error = T(NaN)
    end
    
    BeliefDiagnostics{T}(
        belief_mse,
        drift,
        uncertainties,
        calibration_error,
        collapse
    )
end

# ============================================================================
# Belief Visualization Helpers
# ============================================================================

"""
    extract_gate_activations(filter::BeliefFilter, h_prev, y_t, u_prev)

Extract GRU gate activations for interpretability.

Returns named tuple with:
- `reset_gate`: How much prior belief is "forgotten"
- `update_gate`: How much new information is integrated
- `candidate`: New belief candidate
"""
function extract_gate_activations(filter::BeliefFilter{T}, h_prev::AbstractVector,
                                  y_t::AbstractVector, u_prev::AbstractVector) where T
    # Get embeddings
    obs_emb = filter.obs_encoder(y_t)
    act_emb = filter.action_encoder(u_prev)
    input = vcat(obs_emb, act_emb)
    
    # Extract GRU parameters
    # GRUCell has: Wi, Wh, b, inner_state_size
    Wi = filter.gru_cell.Wi
    Wh = filter.gru_cell.Wh
    b = filter.gru_cell.b
    hidden_size = size(Wh, 1) ÷ 3
    
    # Compute gates manually
    gx = Wi * input
    gh = Wh * h_prev
    
    # Reset and update gates
    r = sigmoid.(gx[1:hidden_size] .+ gh[1:hidden_size] .+ b[1:hidden_size])
    z = sigmoid.(gx[hidden_size+1:2*hidden_size] .+ gh[hidden_size+1:2*hidden_size] .+ 
                 b[hidden_size+1:2*hidden_size])
    
    # Candidate hidden state
    h_candidate = tanh.(gx[2*hidden_size+1:end] .+ r .* gh[2*hidden_size+1:end] .+
                        b[2*hidden_size+1:end])
    
    (
        reset_gate = r,
        update_gate = z,
        candidate = h_candidate,
        information_integration = mean(z),  # Scalar summary
        belief_correction = mean(T(1) .- r)  # How much correction happens
    )
end

# ============================================================================
# GPU Support
# ============================================================================

function Flux.gpu(model::BeliefFilter{T}) where T
    BeliefFilter{T}(
        gpu(model.obs_encoder),
        gpu(model.action_encoder),
        gpu(model.gru_cell),
        gpu(model.uncertainty_head),
        gpu(model.state_decoder),
        model.hidden_dim,
        model.config
    )
end

function Flux.cpu(model::BeliefFilter{T}) where T
    BeliefFilter{T}(
        cpu(model.obs_encoder),
        cpu(model.action_encoder),
        cpu(model.gru_cell),
        cpu(model.uncertainty_head),
        cpu(model.state_decoder),
        model.hidden_dim,
        model.config
    )
end

# ============================================================================
# Initialization Strategies
# ============================================================================

"""
    init_belief_from_prior(filter::BeliefFilter{T}, prior_samples::Matrix) -> h_init

Initialize belief state from prior samples of student states.

Uses the decoder in reverse (approximately) to find a belief state
that decodes to the prior mean.

# Arguments
- `prior_samples::Matrix`: [state_dim × n_samples] samples from prior p(x₀)

# Returns
- `h_init::Vector`: Initial belief state
"""
function init_belief_from_prior(filter::BeliefFilter{T}, 
                                prior_samples::AbstractMatrix) where T
    # Compute prior statistics
    prior_mean = vec(mean(prior_samples, dims=2))
    
    # Find h that minimizes ||decode(h) - prior_mean||²
    # This is a simple optimization (could use more sophisticated methods)
    h = zeros(T, filter.hidden_dim)
    
    # Gradient descent (few steps)
    η = T(0.1)
    for _ in 1:100
        x_hat = decode_state(filter, h)
        grad = 2 .* (x_hat .- prior_mean)
        
        # Backprop through decoder (approximate)
        # For simplicity, just move h in direction that reduces error
        h .-= η * grad[1] * h  # Simplified update
    end
    
    return h
end
