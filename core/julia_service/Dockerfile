FROM julia:1.10

WORKDIR /app

# 1. Install Dependencies
# CRITICAL: We DO NOT copy the local Project.toml because it contains version constraints
# that conflict with the Docker Julia version. We build a FRESH project here.

RUN julia -e 'using Pkg; Pkg.generate("StudentIO")'
WORKDIR /app/StudentIO

# Install Core Dependencies fresh for this environment
RUN julia --project=. -e 'using Pkg; Pkg.add(["Oxygen", "HTTP", "JSON3", "Flux", "Distributions", "LinearAlgebra", "Statistics", "Random", "Zygote"]); Pkg.instantiate(); Pkg.precompile()'

# 2. Copy Source Code
# We overlay the local source code onto the fresh project
COPY src/ src/
COPY server.jl .

# 3. Expose Port
EXPOSE 8080

# 4. Run the Real Server
CMD ["julia", "--project=.", "server.jl"]
