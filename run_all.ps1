# StudentIO Minimal Startup Script
$ErrorActionPreference = "Stop"

Write-Host "Starting StudentIO..." -ForegroundColor Cyan

# 1. Start Python AI Service (Port 8000)
Write-Host "Starting Python (8000)..." -ForegroundColor Magenta
# Check for venv and construct command
$venvPath = "$PWD\.venv\Scripts\Activate.ps1"
$pyCmd = "python main.py"
if (Test-Path $venvPath) {
    # We use & to invoke the script, handling spaces correctly
    $pyCmd = "Write-Host 'Activating Venv...'; & '$venvPath'; python main.py"
}
else {
    $pyCmd = "python main.py"
}

# Append a pause to ensure we see output if it crashes immediately
# Append a pause to ensure we see output if it crashes immediately
$pyCmd = "$pyCmd; if (`$?) { Write-Host 'Server Stopped' } else { Write-Host 'Server Crashed' -ForegroundColor Red }; Read-Host 'Press Enter to close...'"

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", "$pyCmd" -WorkingDirectory "$PWD\core\ai_service"
Start-Sleep -Seconds 2

# 2. Start Node.js Backend API (Port 3000)
Write-Host "Starting Backend (3000)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "node server.js" -WorkingDirectory "$PWD\backend"
Start-Sleep -Seconds 2

# 3. Start Julia Backend (Port 8080)
Write-Host "Starting Julia (8080)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "julia --project=. server.jl" -WorkingDirectory "$PWD\core\julia_service"
Start-Sleep -Seconds 2

# 4. Start React Frontend (Port 5173)
Write-Host "Starting Frontend (5173)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -WorkingDirectory "$PWD\frontend"

Write-Host ""
Write-Host "All services started." -ForegroundColor Green
