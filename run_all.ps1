# StudentIO Minimal Startup Script
$ErrorActionPreference = "Stop"

Write-Host "Starting StudentIO..." -ForegroundColor Cyan

# 1. Start Python AI Service (Port 8000)
Write-Host "Starting Python (8000)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python main.py" -WorkingDirectory "$PWD\backend\ai_service"
Start-Sleep -Seconds 2

# 2. Start Node.js Backend API (Port 3000)
Write-Host "Starting Backend (3000)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "node server.js" -WorkingDirectory "$PWD\backend"
Start-Sleep -Seconds 2

# 3. Start Julia Backend (Port 8080)
Write-Host "Starting Julia (8080)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "julia --project=. -e 'using StudentIO; StudentIO.start_server()'" -WorkingDirectory "$PWD"
Start-Sleep -Seconds 2

# 4. Start React Frontend (Port 5173)
Write-Host "Starting Frontend (5173)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -WorkingDirectory "$PWD"

Write-Host ""
Write-Host "All services started." -ForegroundColor Green
