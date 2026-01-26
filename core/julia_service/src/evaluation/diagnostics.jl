# StudentIO Diagnostics - Part 1: Core Functions

using Statistics
using LinearAlgebra
using Printf

struct DiagnosticResults{T<:AbstractFloat}
    belief_mse::T
    belief_correlation::T
    uncertainty_calibration::T
    adaptation_steps::Int
    retention_score::T
    policy_value::T
    reward_trajectory::Vector{T}
end

function evaluate_belief_calibration(model::StudentIOModel{T}, student::SyntheticStudent{T}, steps::Int=100) where T
    reset!(student)
    h = zeros(T, model.config.belief_dim)
    u_prev = zeros(T, model.config.action_dim)
    
    belief_errors, true_mast, est_mast = T[], T[], T[]
    
    for _ in 1:steps
        action, _ = select_action(model.policy, h)
        obs = step!(student, action)
        x_true = student.true_state
        h, _ = update_belief(model.filter, h, obs, u_prev)
        x_hat = decode_state(model.filter, h)
        
        push!(belief_errors, norm(x_true - x_hat))
        push!(true_mast, mean(x_true[1:model.config.mastery_dim]))
        push!(est_mast, mean(x_hat[1:model.config.mastery_dim]))
        u_prev = encode_action(model.policy, action)
    end
    
    mse = mean(belief_errors.^2)
    corr = length(true_mast) > 2 ? cor(true_mast, est_mast) : zero(T)
    return mse, isnan(corr) ? zero(T) : corr
end

function measure_adaptation_speed(model::StudentIOModel{T}, student::SyntheticStudent{T}; threshold::T=T(0.8), max_steps::Int=500) where T
    reset!(student)
    h = zeros(T, model.config.belief_dim)
    u_prev = zeros(T, model.config.action_dim)
    recent = T[]
    
    for t in 1:max_steps
        action, _ = select_action(model.policy, h)
        obs = step!(student, action)
        h, _ = update_belief(model.filter, h, obs, u_prev)
        x_hat = decode_state(model.filter, h)
        
        err = norm(student.true_state - x_hat) / (norm(student.true_state) + T(1e-8))
        push!(recent, max(zero(T), one(T) - err))
        length(recent) > 10 && popfirst!(recent)
        
        length(recent) >= 10 && mean(recent) >= threshold && return t
        u_prev = encode_action(model.policy, action)
    end
    return max_steps
end

function run_diagnostics(model::StudentIOModel{T}, task_dist::TaskDistribution; n_students::Int=10, steps::Int=100) where T
    cfg = StudentStateConfig(state_dim=model.config.state_dim, mastery_dim=model.config.mastery_dim,
        misconception_dim=model.config.misconception_dim, abstraction_dim=model.config.abstraction_dim,
        action_dim=model.config.action_dim, observation_dim=model.config.observation_dim, belief_dim=model.config.belief_dim)
    
    mse_s, corr_s, adapt_s, val_s = zero(T), zero(T), 0, zero(T)
    all_rewards = T[]
    
    for _ in 1:n_students
        task = sample_student(task_dist)
        stud = SyntheticStudent(task, cfg)
        mse, corr = evaluate_belief_calibration(model, stud, steps)
        mse_s += mse; corr_s += corr
        reset!(stud); adapt_s += measure_adaptation_speed(model, stud)
        
        reset!(stud)
        h = zeros(T, model.config.belief_dim)
        for _ in 1:steps
            val_s += value_estimate(model.policy, h)
            action, _ = select_action(model.policy, h)
            obs = step!(stud, action)
            x_prev = length(stud.history) > 1 ? stud.history[end-1].next_state : task.prior_knowledge
            r, _ = compute_reward(model.reward, x_prev, stud.true_state, action; mastery_dim=cfg.mastery_dim, misconception_dim=cfg.misconception_dim)
            push!(all_rewards, r)
            h, _ = update_belief(model.filter, h, obs, zeros(T, cfg.action_dim))
        end
    end
    
    n = T(n_students)
    DiagnosticResults{T}(mse_s/n, corr_s/n, zero(T), adapt_s÷n_students, one(T), val_s/(n*T(steps)), all_rewards)
end

function print_diagnostics(r::DiagnosticResults{T}) where T
    println("="^50, "\nStudentIO Diagnostics\n", "="^50)
    @printf("Belief MSE: %.4f | Correlation: %.4f\n", r.belief_mse, r.belief_correlation)
    @printf("Adaptation Steps: %d | Mean Reward: %.4f\n", r.adaptation_steps, mean(r.reward_trajectory))
end
