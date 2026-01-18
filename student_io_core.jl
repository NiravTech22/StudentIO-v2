using Oxygen, HTTP, JSON3, Libdl, LinearAlgebra, CUDA

#memory mapped structure
struct StudentState_C
    knowledge::NTuple{50, Float32}
    belief_mean::NTuple{50, Float32}
    belief_var::NTuple{50, Float32}
    hidden::NTuple{128, Float32}
    timestep::Int32
    learning_rate::Float32
    forgetting_rate::Float32
end

# Load Shared Library
const LIB_PATH = "./libstudentio.so"
const student_lib = dlopen(LIB_PATH)
const run_step_ptr = dlsym(student_lib, :runTimestep)

#GPU Buffer
const BATCH_SIZE = 32
d_states = CUDA.zeros(StudentState_C, BATCH_SIZE)
d_obs = CUDA.zeros(Float32, BATCH_SIZE * 24) # Simplified size
d_actions = CUDA.zeros(Float32, BATCH_SIZE * 13)
p
@get "/api/telemetry" function(req::HTTP.Request)
    # Trigger the CUDA Kernels
    ccall(run_step_ptr, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32), 
          d_states, d_obs, d_actions, Int32(BATCH_SIZE))
    
    # Copy data back for the UI
    h_states = Array(d_states)
    
    return Dict(
        "timestamp" => time(),
        "students" => [
            Dict(
                "id" => i,
                "belief" => h_states[i].belief_mean,
                "variance" => h_states[i].belief_var,
                "timestep" => h_states[i].timestep
            ) for i in 1:5 # Just return first 5 for UI performance
        ]
    )
end

println("StudentIO Julia Server running on port 8080...")
serve(port=8080)