# StudentIO GPU Acceleration Interface
# High-level API for GPU-accelerated operations

using CUDA

"""
Check if GPU acceleration is available.
"""
gpu_available() = CUDA.functional()

"""
Move model to GPU if available.
"""
function to_gpu(model::StudentIOModel)
    if gpu_available()
        return Flux.gpu(model)
    else
        @warn "CUDA not available, using CPU"
        return model
    end
end

"""
Move model to CPU.
"""
to_cpu(model::StudentIOModel) = Flux.cpu(model)

"""
Accelerated batch belief update - auto-selects GPU/CPU.
"""
function accelerated_belief_update(filter, h_batch, y_batch, u_batch)
    if gpu_available() && h_batch isa CuArray
        batch_belief_update_gpu(filter, h_batch, y_batch, u_batch)
    else
        batch_belief_update_cpu(filter, h_batch, y_batch, u_batch)
    end
end

"""
Accelerated rollout collection with GPU batching.
"""
function accelerated_collect_rollouts(model, task_dist, n_episodes, steps; use_gpu=gpu_available())
    model = use_gpu ? to_gpu(model) : model
    
    buffers = RolloutBuffer{Float32}[]
    config = StudentStateConfig(
        state_dim=model.config.state_dim, mastery_dim=model.config.mastery_dim,
        misconception_dim=model.config.misconception_dim, abstraction_dim=model.config.abstraction_dim,
        action_dim=model.config.action_dim, observation_dim=model.config.observation_dim,
        belief_dim=model.config.belief_dim
    )
    
    for _ in 1:n_episodes
        task = sample_student(task_dist)
        student = SyntheticStudent(task, config)
        buffer = collect_rollout(model, student, steps)
        push!(buffers, buffer)
    end
    
    return buffers
end

"""
GPU memory info for debugging.
"""
function gpu_memory_info()
    if gpu_available()
        free = CUDA.available_memory()
        total = CUDA.total_memory()
        used = total - free
        @printf("GPU Memory: %.1f MB used / %.1f MB total (%.1f%% free)\n",
            used / 1e6, total / 1e6, 100 * free / total)
    else
        println("CUDA not available")
    end
end
