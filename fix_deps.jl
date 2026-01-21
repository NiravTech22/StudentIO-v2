using Pkg
println("Activating project...")
Pkg.activate(".")
println("Adding missing dependencies...")
Pkg.add(["Printf", "Oxygen", "HTTP", "JSON3", "CUDA"])
println("Resolving and Instantiating...")
Pkg.resolve()
Pkg.instantiate()
println("Dependencies successfully installed and environment ready!")
