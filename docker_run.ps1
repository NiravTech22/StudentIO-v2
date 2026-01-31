
# Docker Run Script for StudentIO
# Usage: ./docker_run.ps1
$ErrorActionPreference = "Stop"

Write-Host "Stopping any existing containers..." -ForegroundColor Yellow
docker-compose down

Write-Host "Building and starting StudentIO via Docker..." -ForegroundColor Cyan
docker-compose up --build -d

Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "Containers Status:" -ForegroundColor Green
docker-compose ps

Write-Host ""
Write-Host "StudentIO should be running at http://localhost:5173" -ForegroundColor Green
Write-Host "To view logs, run: docker-compose logs -f"
