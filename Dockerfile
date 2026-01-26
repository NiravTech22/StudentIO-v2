# Use official Julia image
FROM julia:1.9

# Set working directory
WORKDIR /app

# Copy project definition files
COPY Project.toml Manifest.toml ./

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy source code
COPY src/ ./src/

# Expose the API port
EXPOSE 8080

# Run the server using the module entry point
CMD ["julia", "--project=.", "-e", "using StudentIO; StudentIO.start_server(host=\"0.0.0.0\", port=8080)"]
