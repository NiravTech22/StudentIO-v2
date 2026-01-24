# Minimal test of Julia loading
println("Loading core types...")
try
    include("src/core/types.jl")
    println("✓ Types loaded")
catch e
    println("ERROR in types.jl:")
    showerror(stdout, e)
    println()
end

println("\nLoading latent state...")
try
    include("src/core/latent_state.jl")
    println("✓ Latent state loaded")
catch e
    println("ERROR in latent_state.jl:")
    showerror(stdout, e)
    println()
end

println("\nLoading observation model...")
try
    include("src/core/observation_model.jl")
    println("✓ Observation model loaded")
catch e
    println("ERROR in observation_model.jl:")
    showerror(stdout, e)
    println()
end

println("\nLoading belief filter...")
try
    include("src/core/belief_filter.jl")
    println("✓ Belief filter loaded")
catch e
    println("ERROR in belief_filter.jl:")
    showerror(stdout, e)
    println()
end

println("\nLoading policy...")
try
    include("src/core/policy.jl")
    println("✓ Policy loaded")
catch e
    println("ERROR in policy.jl:")
    showerror(stdout, e)
    println()
end

println("\nLoading reward...")
try
    include("src/core/reward.jl")
    println("✓ Reward loaded")
catch e
    println("ERROR in reward.jl:")
    showerror(stdout, e)
    println()
end

println("\n=== Core modules test complete ===")
