try
    println("Attempting to include src/StudentIO.jl")
    include("src/StudentIO.jl")
    println("Success!")
catch e
    println("CAUGHT ERROR:")
    showerror(stdout, e)
    println()
    # Print stacktrace
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
