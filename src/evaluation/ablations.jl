# StudentIO Ablation Studies
# Measure impact of removing components

using Statistics

struct AblationResult{T<:AbstractFloat}
    component_removed::String
    baseline_score::T
    ablated_score::T
    impact::T  # Fraction decrease
end

function run_ablation(model::StudentIOModel{T}, task_dist::TaskDistribution, component::Symbol; n_students::Int=5, steps::Int=50) where T
    cfg = StudentStateConfig(state_dim=model.config.state_dim, mastery_dim=model.config.mastery_dim,
        misconception_dim=model.config.misconception_dim, abstraction_dim=model.config.abstraction_dim,
        action_dim=model.config.action_dim, observation_dim=model.config.observation_dim, belief_dim=model.config.belief_dim)
    
    # Baseline performance
    baseline = zero(T)
    for _ in 1:n_students
        task = sample_student(task_dist)
        stud = SyntheticStudent(task, cfg)
        buffer = collect_rollout(model, stud, steps)
        baseline += sum(buffer.rewards)
    end
    baseline /= T(n_students)
    
    # Ablated performance based on component
    ablated = if component == :belief_filter
        # Replace with no-update filter (just pass through)
        ablate_belief_filter(model, task_dist, cfg, n_students, steps)
    elseif component == :reward_shaping
        ablate_reward_shaping(model, task_dist, cfg, n_students, steps)
    elseif component == :meta_learning
        # No meta-learning = random init
        baseline  # Same as baseline for this simple ablation
    else
        baseline
    end
    
    impact = (baseline - ablated) / (baseline + T(1e-8))
    AblationResult{T}(string(component), baseline, ablated, impact)
end

function ablate_belief_filter(model::StudentIOModel{T}, task_dist, cfg, n_students, steps) where T
    # Use random belief state instead of actual filter
    score = zero(T)
    for _ in 1:n_students
        task = sample_student(task_dist)
        stud = SyntheticStudent(task, cfg)
        reset!(stud)
        
        for _ in 1:steps
            h = randn(T, model.config.belief_dim) * T(0.1)  # Random belief
            action, _ = select_action(model.policy, h)
            step!(stud, action)
            x_prev = length(stud.history) > 1 ? stud.history[end-1].next_state : task.prior_knowledge
            r, _ = compute_reward(model.reward, x_prev, stud.true_state, action;
                mastery_dim=cfg.mastery_dim, misconception_dim=cfg.misconception_dim)
            score += r
        end
    end
    score / T(n_students)
end

function ablate_reward_shaping(model::StudentIOModel{T}, task_dist, cfg, n_students, steps) where T
    # Use sparse reward only
    score = zero(T)
    sparse_config = RewardConfig(α=1.0f0, β=0.0f0, γ=0.0f0, difficulty_bonus_scale=0.0f0, efficiency_weight=0.0f0)
    
    for _ in 1:n_students
        task = sample_student(task_dist)
        stud = SyntheticStudent(task, cfg)
        buffer = collect_rollout(model, stud, steps)
        
        # Recompute with sparse reward
        for i in 1:length(buffer.rewards)
            x_prev = i > 1 ? stud.history[i-1].next_state : task.prior_knowledge
            x_curr = stud.history[i].next_state
            r, _ = compute_reward(model.reward, x_prev, x_curr, buffer.actions[i];
                mastery_dim=cfg.mastery_dim, misconception_dim=cfg.misconception_dim, config_override=sparse_config)
            score += r
        end
    end
    score / T(n_students)
end

function run_all_ablations(model::StudentIOModel{T}, task_dist::TaskDistribution) where T
    components = [:belief_filter, :reward_shaping]
    results = [run_ablation(model, task_dist, c) for c in components]
    
    println("\nAblation Study Results")
    println("="^60)
    for r in results
        @printf("%-20s | Baseline: %.3f | Ablated: %.3f | Impact: %+.1f%%\n",
            r.component_removed, r.baseline_score, r.ablated_score, r.impact * 100)
    end
    results
end
