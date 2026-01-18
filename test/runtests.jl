# StudentIO Test Suite

using Test
using StudentIO

@testset "StudentIO Tests" begin
    
    @testset "Core Types" begin
        config = StudentStateConfig()
        @test config.state_dim == 64
        @test config.mastery_dim + config.misconception_dim + config.abstraction_dim == config.state_dim
        
        state = StudentState(config)
        @test length(to_vector(state)) == config.state_dim
        
        # Action space
        space = ActionSpace()
        @test space.num_action_types > 0
    end
    
    @testset "Latent State Transition" begin
        config = StudentStateConfig()
        model = TransitionModel{Float32}(config)
        
        x = randn(Float32, config.state_dim)
        u = randn(Float32, config.action_dim)
        
        # Deterministic transition
        x_next = transition(model, x, u; deterministic=true)
        @test length(x_next) == config.state_dim
        
        # Stochastic transition
        x_next_stoch = transition(model, x, u; deterministic=false)
        @test length(x_next_stoch) == config.state_dim
        
        # Batch transition
        x_batch = randn(Float32, config.state_dim, 8)
        u_batch = randn(Float32, config.action_dim, 8)
        x_next_batch = transition_batch(model, x_batch, u_batch)
        @test size(x_next_batch) == (config.state_dim, 8)
    end
    
    @testset "Observation Model" begin
        config = StudentStateConfig()
        model = ObservationModel{Float32}(config)
        
        x = rand(Float32, config.state_dim)
        
        # Observe
        μ, σ = observe(model, x)
        @test length(μ) == config.observation_dim
        @test all(σ .> 0)
        
        # Sample observation
        y = sample_observation(model, x)
        @test length(y) == config.observation_dim
        
        # Log-likelihood
        ll = log_likelihood(model, x, y)
        @test isfinite(ll)
    end
    
    @testset "Belief Filter" begin
        config = StudentStateConfig()
        filter = BeliefFilter{Float32}(config)
        
        h = zeros(Float32, config.belief_dim)
        y = randn(Float32, config.observation_dim)
        u = randn(Float32, config.action_dim)
        
        # Update belief
        h_new, uncertainty = update_belief(filter, h, y, u)
        @test length(h_new) == config.belief_dim
        @test uncertainty > 0
        
        # Decode state
        x_hat = decode_state(filter, h_new)
        @test length(x_hat) == config.state_dim
        
        # Sequence update
        observations = [randn(Float32, config.observation_dim) for _ in 1:10]
        actions = [randn(Float32, config.action_dim) for _ in 1:10]
        traj, uncerts = update_belief_sequence(filter, observations, actions)
        @test size(traj) == (config.belief_dim, 10)
        @test length(uncerts) == 10
    end
    
    @testset "Policy Network" begin
        config = StudentStateConfig()
        policy = PolicyNetwork{Float32}(config)
        
        h = randn(Float32, config.belief_dim)
        
        # Select action
        action, log_p = select_action(policy, h)
        @test haskey(action, :action_type)
        @test haskey(action, :difficulty)
        @test isfinite(log_p)
        
        # Value estimate
        value = value_estimate(policy, h)
        @test isfinite(value)
        
        # Entropy
        ent = action_entropy(policy, h)
        @test ent >= 0
        
        # Explain action
        rationale = explain_action(policy, h, action)
        @test !isempty(rationale.top_belief_dims)
    end
    
    @testset "Reward System" begin
        config = RewardConfig()
        rf = RewardFunction{Float32}()
        
        x_prev = rand(Float32, 64)
        x_curr = rand(Float32, 64)
        x_curr[1:40] .+= 0.1  # Simulate learning
        
        action = (difficulty=0.5f0, pacing=0.5f0, emphasis=0.5f0)
        
        reward, components = compute_reward(rf, x_prev, x_curr, action)
        @test isfinite(reward)
        @test haskey(components, :gain)
        @test haskey(components, :retention)
        
        # GAE computation
        rewards = rand(Float32, 20)
        values = rand(Float32, 20)
        adv, ret = compute_gae(rewards, values, 0.99f0, 0.95f0)
        @test length(adv) == 20
        @test length(ret) == 20
    end
    
    @testset "Task Distribution" begin
        dist = TaskDistribution()
        
        # Sample student
        task = sample_student(dist)
        @test 0 < task.learning_rate < 1
        @test 0 < task.forgetting_rate < 1
        @test length(task.prior_knowledge) > 0
        
        # Sample multiple
        students = sample_students(dist, 5)
        @test length(students) == 5
        
        # Task difficulty
        diff = task_difficulty(task)
        @test diff > 0
    end
    
    @testset "Synthetic Student" begin
        config = StudentStateConfig()
        task = sample_student(TaskDistribution())
        student = SyntheticStudent(task, config)
        
        # Initial state
        @test length(student.true_state) == config.state_dim
        @test student.step_count == 0
        
        # Step
        action = (action_type=PRESENT_PROBLEM, problem_id=1, topic_id=1,
                  difficulty=0.5f0, pacing=0.5f0, emphasis=0.5f0)
        obs = step!(student, action)
        @test length(obs) == config.observation_dim
        @test student.step_count == 1
        
        # Reset
        reset!(student)
        @test student.step_count == 0
        @test isempty(student.history)
    end
    
    @testset "Full Model Integration" begin
        model = create_default_model()
        
        @test model.config.state_dim == 64
        @test model.config.belief_dim == 128
        
        # Create session
        session = create_session(model)
        @test session.step_count == 0
        @test session.uncertainty == 1.0
        
        # Step with observation
        obs = randn(Float32, model.config.observation_dim)
        action = step!(session, obs)
        @test session.step_count == 1
        @test haskey(action, :action_type)
        
        # Reset
        reset!(session)
        @test session.step_count == 0
    end
    
    @testset "Training Loop Components" begin
        config = PPOConfig()
        @test config.clip_ε > 0
        @test config.γ > 0 && config.γ <= 1
        
        # MetaLearner config
        meta_config = MetaLearnerConfig()
        @test meta_config.inner_steps > 0
        @test meta_config.meta_batch_size > 0
    end
    
end

println("All tests passed!")
