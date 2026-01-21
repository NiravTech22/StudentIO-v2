using Pkg
Pkg.activate(".")
Pkg.add("Flux")
Pkg.resolve()
Pkg.instantiate()
println("Flux installed successfully.")
