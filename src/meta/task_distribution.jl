# ============================================================================
# StudentIO Task Distribution
# ============================================================================
#
# This module models the distribution of student learning dynamics.
# Each student is treated as a "task" with unique characteristics.
#
# Key Principle:
#   Students vary in:
#   - Learning rates (how fast they acquire knowledge)
#   - Forgetting rates (how fast knowledge decays)
#   - Noise levels (response variability)
#   - Prior knowledge (initial state x₀)
#   - Response styles (careful, impulsive, guessing)
#
# This enables meta-learning: optimize for fast adaptation to new students.
#
# ============================================================================

using Distributions
using Random
using LinearAlgebra

"""
    ResponseStyle

Categorization of student response behaviors.

- `CAREFUL`: Takes time, high accuracy when learned
- `IMPULSIVE`: Fast responses, noisy accuracy
- `GUESSING`: Responses semi-random, ignores knowledge
- `STRATEGIC`: Adjusts effort based on perceived difficulty
- `FATIGUED`: Performance degrades over time
"""
@enum ResponseStyle begin
    CAREFUL
    IMPULSIVE
    GUESSING
    STRATEGIC
    FATIGUED
end

"""
    StudentTask{T<:AbstractFloat}

Parameters defining a single student's learning dynamics.

Each student is a unique "task" in the meta-learning sense.
These parameters control how the synthetic student transitions states
and generates observations.

# Fields
- `learning_rate::T`: Speed of knowledge acquisition [0.05, 0.5]
- `forgetting_rate::T`: Memory decay rate [0.01, 0.2]
- `noise_scale::T`: Response variability [0.1, 1.0]
- `prior_knowledge::Vector{T}`: Initial state x₀
- `response_style::ResponseStyle`: Behavioral pattern
- `attention_span::Int`: Steps before fatigue onset
- `topic_affinities::Vector{T}`: Topic-specific learning modifiers

# Usage
```julia
task = sample_student(TaskDistribution())
student = SyntheticStudent(task)
```
"""
struct StudentTask{T<:AbstractFloat}
    learning_rate::T
    forgetting_rate::T
    noise_scale::T
    prior_knowledge::Vector{T}
    response_style::ResponseStyle
    attention_span::Int
    topic_affinities::Vector{T}
end

"""
    TaskDistribution{T<:AbstractFloat}

Distribution over student tasks for meta-learning.

Samples from this distribution generate diverse students for training.

# Fields
- `learning_rate_dist::LogNormal`: LN(μ=-1.5, σ=0.5) → median ~0.22
- `forgetting_rate_dist::Beta`: Beta(2, 8) → most students forget slowly
- `noise_scale_dist::Gamma`: Gamma(2, 0.15) → mean ~0.3
- `prior_knowledge_mean::Vector{T}`: Population prior mean
- `prior_knowledge_cov::Matrix{T}`: Population prior covariance
- `response_style_probs::Vector{T}`: Probabilities for each style
- `attention_span_dist::Poisson`: Mean ~50 steps
- `topic_affinity_dist::Normal`: N(1, 0.2) per topic

# Design Choice
Distributions are chosen based on educational psychology research:
- Learning rates: Log-normal (positive, wide range)
- Forgetting: Beta (bounded [0,1], right-skewed)
- Noise: Gamma (positive, moderate variance)
"""
struct TaskDistribution{T<:AbstractFloat}
    learning_rate_dist::LogNormal{T}
    forgetting_rate_dist::Beta{T}
    noise_scale_dist::Gamma{T}
    prior_knowledge_mean::Vector{T}
    prior_knowledge_cov::Matrix{T}
    response_style_probs::Vector{T}
    attention_span_dist::Poisson{T}
    topic_affinity_scale::T
    state_dim::Int
    mastery_dim::Int
end

"""
    TaskDistribution{T}(; config, state_dim=64, mastery_dim=40) where T

Construct default task distribution.

# Keyword Arguments
- `state_dim::Int=64`: Latent state dimension
- `mastery_dim::Int=40`: Mastery dimensions
- `learning_rate_params::Tuple=(−1.5, 0.5)`: LogNormal parameters
- `forgetting_rate_params::Tuple=(2.0, 8.0)`: Beta parameters
- `response_style_probs::Vector`: Probability of each style
"""
function TaskDistribution{T}(;
    state_dim::Int = 64,
    mastery_dim::Int = 40,
    learning_rate_params::Tuple = (-1.5, 0.5),
    forgetting_rate_params::Tuple = (2.0, 8.0),
    noise_scale_params::Tuple = (2.0, 0.15),
    response_style_probs::Vector = [0.4, 0.25, 0.1, 0.15, 0.1],
    attention_span_mean::Real = 50.0,
    topic_affinity_scale::Real = 0.2
) where T<:AbstractFloat
    
    @assert length(response_style_probs) == 5 "Must have 5 response style probabilities"
    @assert sum(response_style_probs) ≈ 1.0 "Style probabilities must sum to 1"
    
    # Distributions
    lr_dist = LogNormal(T(learning_rate_params[1]), T(learning_rate_params[2]))
    fr_dist = Beta(T(forgetting_rate_params[1]), T(forgetting_rate_params[2]))
    ns_dist = Gamma(T(noise_scale_params[1]), T(noise_scale_params[2]))
    as_dist = Poisson(T(attention_span_mean))
    
    # Prior knowledge distribution (low initial mastery, no misconceptions)
    prior_mean = zeros(T, state_dim)
    prior_mean[1:mastery_dim] .= T(0.1)  # Low initial mastery
    
    # Prior covariance (diagonal, more variance in mastery)
    prior_var = fill(T(0.05), state_dim)
    prior_var[1:mastery_dim] .= T(0.1)  # More variance in mastery
    prior_cov = Diagonal(prior_var)
    
    TaskDistribution{T}(
        lr_dist,
        fr_dist,
        ns_dist,
        prior_mean,
        Matrix(prior_cov),
        convert(Vector{T}, response_style_probs),
        as_dist,
        T(topic_affinity_scale),
        state_dim,
        mastery_dim
    )
end

# Default constructor
TaskDistribution() = TaskDistribution{Float32}()

"""
    sample_student(dist::TaskDistribution{T}) -> StudentTask{T}

Sample a new student task from the distribution.

Each call generates a unique student with different learning dynamics.
"""
function sample_student(dist::TaskDistribution{T}) where T
    # Sample base parameters
    learning_rate = T(rand(dist.learning_rate_dist))
    learning_rate = clamp(learning_rate, T(0.01), T(1.0))  # Bound
    
    forgetting_rate = T(rand(dist.forgetting_rate_dist))
    forgetting_rate = clamp(forgetting_rate, T(0.001), T(0.5))
    
    noise_scale = T(rand(dist.noise_scale_dist))
    noise_scale = clamp(noise_scale, T(0.05), T(2.0))
    
    # Sample prior knowledge
    prior_dist = MvNormal(dist.prior_knowledge_mean, dist.prior_knowledge_cov)
    prior_knowledge = T.(rand(prior_dist))
    
    # Clamp mastery and misconceptions to valid range
    prior_knowledge[1:dist.mastery_dim] = clamp.(prior_knowledge[1:dist.mastery_dim], T(0), T(1))
    
    # Sample response style
    style_idx = sample_categorical_dist(dist.response_style_probs)
    response_style = ResponseStyle(style_idx - 1)
    
    # Sample attention span
    attention_span = max(10, Int(rand(dist.attention_span_dist)))
    
    # Sample topic affinities
    topic_affinities = T(1.0) .+ T(dist.topic_affinity_scale) .* randn(T, dist.mastery_dim)
    topic_affinities = clamp.(topic_affinities, T(0.5), T(2.0))
    
    StudentTask{T}(
        learning_rate,
        forgetting_rate,
        noise_scale,
        prior_knowledge,
        response_style,
        attention_span,
        topic_affinities
    )
end

"""
    sample_categorical_dist(probs::AbstractVector) -> Int

Sample from categorical distribution.
"""
function sample_categorical_dist(probs::AbstractVector{T}) where T
    r = rand(T)
    cumsum = zero(T)
    for i in eachindex(probs)
        cumsum += probs[i]
        if r <= cumsum
            return i
        end
    end
    return length(probs)
end

"""
    sample_students(dist::TaskDistribution, n::Int) -> Vector{StudentTask}

Sample n students from the distribution.
"""
function sample_students(dist::TaskDistribution{T}, n::Int) where T
    [sample_student(dist) for _ in 1:n]
end

# ============================================================================
# Task Modification (Ablations)
# ============================================================================

"""
    modify_task(task::StudentTask; kwargs...) -> StudentTask

Create a modified copy of a student task.

Useful for ablation studies.
"""
function modify_task(task::StudentTask{T};
                     learning_rate::Union{Nothing, Real} = nothing,
                     forgetting_rate::Union{Nothing, Real} = nothing,
                     noise_scale::Union{Nothing, Real} = nothing,
                     response_style::Union{Nothing, ResponseStyle} = nothing) where T
    StudentTask{T}(
        isnothing(learning_rate) ? task.learning_rate : T(learning_rate),
        isnothing(forgetting_rate) ? task.forgetting_rate : T(forgetting_rate),
        isnothing(noise_scale) ? task.noise_scale : T(noise_scale),
        task.prior_knowledge,
        isnothing(response_style) ? task.response_style : response_style,
        task.attention_span,
        task.topic_affinities
    )
end

# ============================================================================
# Task Analysis
# ============================================================================

"""
    task_difficulty(task::StudentTask) -> Real

Estimate overall difficulty of a student task.

Higher = harder to teach (slow learner, fast forgetter, high noise).
"""
function task_difficulty(task::StudentTask{T}) where T
    # Harder students: slow learning, fast forgetting, high noise
    lr_factor = T(1.0) / (task.learning_rate + T(0.01))
    fr_factor = task.forgetting_rate * T(10.0)
    noise_factor = task.noise_scale
    
    return lr_factor + fr_factor + noise_factor
end

"""
    task_similarity(task1::StudentTask, task2::StudentTask) -> Real

Compute similarity between two student tasks.

Used for task clustering in meta-learning.
"""
function task_similarity(task1::StudentTask{T}, task2::StudentTask{T}) where T
    lr_diff = abs(task1.learning_rate - task2.learning_rate)
    fr_diff = abs(task1.forgetting_rate - task2.forgetting_rate)
    ns_diff = abs(task1.noise_scale - task2.noise_scale)
    
    # Prior knowledge cosine similarity
    prior_sim = dot(task1.prior_knowledge, task2.prior_knowledge) /
                (norm(task1.prior_knowledge) * norm(task2.prior_knowledge) + T(1e-8))
    
    # Response style match
    style_match = task1.response_style == task2.response_style ? T(1.0) : T(0.0)
    
    # Combine (higher = more similar)
    return prior_sim * (T(1.0) - lr_diff) * (T(1.0) - fr_diff) + T(0.2) * style_match
end

"""
    describe_task(task::StudentTask) -> String

Generate human-readable description of a student task.
"""
function describe_task(task::StudentTask)
    lr_desc = if task.learning_rate > 0.3
        "fast learner"
    elseif task.learning_rate > 0.1
        "average learner"
    else
        "slow learner"
    end
    
    fr_desc = if task.forgetting_rate > 0.1
        "forgets quickly"
    elseif task.forgetting_rate > 0.03
        "normal retention"
    else
        "excellent retention"
    end
    
    noise_desc = if task.noise_scale > 0.5
        "very noisy responses"
    elseif task.noise_scale > 0.2
        "moderate response noise"
    else
        "consistent responses"
    end
    
    style_desc = string(task.response_style)
    
    avg_mastery = mean(task.prior_knowledge[1:min(40, length(task.prior_knowledge))])
    prior_desc = if avg_mastery > 0.3
        "high prior knowledge"
    elseif avg_mastery > 0.1
        "moderate prior knowledge"
    else
        "low prior knowledge"
    end
    
    return "Student: $lr_desc, $fr_desc, $noise_desc, $style_desc style, $prior_desc"
end

# ============================================================================
# Stratified Sampling for Balanced Training
# ============================================================================

"""
    sample_stratified(dist::TaskDistribution, n_per_stratum::Int) -> Vector{StudentTask}

Sample students stratified by response style and difficulty.

Ensures training data covers the full task distribution.
"""
function sample_stratified(dist::TaskDistribution{T}, n_per_stratum::Int) where T
    students = StudentTask{T}[]
    
    for style in instances(ResponseStyle)
        for _ in 1:n_per_stratum
            # Sample and modify to have specific style
            task = sample_student(dist)
            modified = StudentTask{T}(
                task.learning_rate,
                task.forgetting_rate,
                task.noise_scale,
                task.prior_knowledge,
                style,
                task.attention_span,
                task.topic_affinities
            )
            push!(students, modified)
        end
    end
    
    return students
end
