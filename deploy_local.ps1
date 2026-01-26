# StudentIO Local Deployment script
Write-Host "Starting Docker Deployment..."

# Stop existing containers
docker-compose down

# Build and Start
docker-compose up -d --build

Write-Host "Deployment Complete!"
Write-Host "Frontend: http://localhost"
Write-Host "API:      http://localhost:3000"
Write-Host "AI Core:  http://localhost:8000"
Write-Host " "
Write-Host "To view logs run:"
Write-Host "docker-compose logs -f"
