using Oxygen, HTTP, JSON3, Libdl, LinearAlgebra, CUDA, Random

# ============================================================================
# Types
# ============================================================================

# Memory mapped structure (Matches C definition)
struct StudentState_C
    knowledge::NTuple{50, Float32}
    belief_mean::NTuple{50, Float32}
    belief_var::NTuple{50, Float32}
    hidden::NTuple{128, Float32}
    timestep::Int32
    learning_rate::Float32
    forgetting_rate::Float32
end

# Mutable struct for Julia-side simulation (Fallback)
mutable struct StudentState_Sim
    id::Int
    belief_mean::Vector{Float32}
    belief_var::Vector{Float32}
    timestep::Int
    
    function StudentState_Sim(id::Int)
        new(id, rand(Float32, 50), ones(Float32, 50) .* 0.5f0, 0)
    end
end

# ============================================================================
# Native Library Loading
# ============================================================================

const LIB_NAME = Sys.iswindows() ? "student_io_core.dll" : "libstudentio.so"
const LIB_PATH = joinpath(@__DIR__, LIB_NAME)

global use_native = false
global run_step_ptr = C_NULL
global d_states = nothing
global d_obs = nothing
global d_actions = nothing
global BATCH_SIZE = 5

try
    if isfile(LIB_PATH)
        println("Found native library: $LIB_PATH")
        student_lib = dlopen(LIB_PATH)
        global run_step_ptr = dlsym(student_lib, :runTimestep)
        
        # Initialize GPU buffers if library loaded successfully
        global BATCH_SIZE = 32
        global d_states = CUDA.zeros(StudentState_C, BATCH_SIZE)
        global d_obs = CUDA.zeros(Float32, BATCH_SIZE * 24)
        global d_actions = CUDA.zeros(Float32, BATCH_SIZE * 13)
        
        println("CUDA Backend Initialized Successfully.")
        global use_native = true
    else
        println("Native library not found at $LIB_PATH. Switching to SIMULATION MODE.")
    end
catch e
    println("Failed to load native library or initialize CUDA: $e")
    println("Switching to SIMULATION MODE.")
    global use_native = false
end

# ============================================================================
# Simulation Mode State
# ============================================================================

const sim_students = [StudentState_Sim(i) for i in 1:5]

function run_simulation_step!()
    for s in sim_students
        # Drifting random walk for belief
        noise = randn(Float32, 50) .* 0.05f0
        s.belief_mean .= clamp.(s.belief_mean .+ noise, 0.0f0, 1.0f0)
        
        # Varying variance
        var_noise = randn(Float32, 50) .* 0.01f0
        s.belief_var .= clamp.(s.belief_var .+ var_noise, 0.0f0, 1.0f0)
        
        s.timestep += 1
    end
end

# ============================================================================
# API Endpoints
# ============================================================================

@get "/api/telemetry" function(req::HTTP.Request)
    response_data = []

    if use_native
        try
            # Trigger CUDA Kernels
            ccall(run_step_ptr, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32), 
                  d_states, d_obs, d_actions, Int32(BATCH_SIZE))
            
            # Copy back to CPU
            h_states = Array(d_states)
            
            # Format for UI
            response_data = [
                Dict(
                    "id" => i,
                    "belief" => collect(h_states[i].belief_mean),
                    "variance" => collect(h_states[i].belief_var),
                    "timestep" => h_states[i].timestep
                ) for i in 1:5 # Limit to 5 for dashboard
            ]
        catch e
            println("Runtime CUDA Error: $e")
            # Fallback if runtime fails
            run_simulation_step!()
            response_data = [
                Dict(
                    "id" => s.id,
                    "belief" => s.belief_mean,
                    "variance" => s.belief_var,
                    "timestep" => s.timestep
                ) for s in sim_students
            ]
        end
    else
        # Pure Julia Simulation
        run_simulation_step!()
        response_data = [
            Dict(
                "id" => s.id,
                "belief" => s.belief_mean,
                "variance" => s.belief_var,
                "timestep" => s.timestep
            ) for s in sim_students
        ]
    end

    response_body = Dict(
        "timestamp" => time(),
        "students" => response_data,
        "mode" => use_native ? "CUDA" : "SIMULATION"
    )
    
    return HTTP.Response(200, ["Access-Control-Allow-Origin" => "*", "Content-Type" => "application/json"], JSON3.write(response_body))
end

# Removed unstable internal hack

println("StudentIO Julia Server running on port 8080...")
println("Mode: ", use_native ? "CUDA ACCELERATED" : "CPU SIMULATION")
serve(port=8080)