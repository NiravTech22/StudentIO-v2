println("Testing StudentIO.jl loading step by step...")

# Test 1: Load types
println("\n1. Loading core/types.jl...")
try
    include("src/core/types.jl")
    println("   ✓ types.jl loaded")
catch e
    println("   ✗ ERROR:")
    showerror(stdout, e)
    println()
    exit(1)
end

# Test 2: Check if we can define module  
println("\n2. Test module definition...")
try
    module TestModule
    include("src/core/types.jl")
    end
    println("   ✓ Module wrapping works")
catch e
    println("   ✗ ERROR:")
    showerror(stdout, e)
    println()
    exit(1)
end

# Test 3: Load with Flux imports
println("\n3. Loading with Flux context...")
try
    module TestModule2
    using Flux
    using Flux: Chain, Dense, GRU, softmax, sigmoid, relu
    using Zygote
    using Distributions
    using LinearAlgebra
    using Random
    using Statistics

    include("src/core/types.jl")
    include("src/core/latent_state.jl")
    include("src/core/observation_model.jl")
    include("src/core/belief_filter.jl")
    include("src/core/policy.jl")
    include("src/core/reward.jl")
    end
    println("   ✓ All core modules loaded in module context")
catch e
    println("   ✗ ERROR:")
    showerror(stdout, e)
    println()
    exit(1)
end

println("\n✓✓✓ All tests passed!")
