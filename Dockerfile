# Use official Julia image
FROM julia:1.10

# Set working directory
WORKDIR /app

# Copy project definition files
COPY Project.toml Manifest.toml ./

# Install dependencies
# We use a build script to precompile and ensure packages are ready
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy source code
COPY src/ ./src/
COPY student_io_core.jl .

# Expose the API port
EXPOSE 8080

# Run the server
# We use --compiled-modules=no as a safety for container environments if precompile fails,
# but ideally we want precompilation. For now, matching the local workaound.
CMD ["julia", "--project=.", "--compiled-modules=no", "student_io_core.jl"]
