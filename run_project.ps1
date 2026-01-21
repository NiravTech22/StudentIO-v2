$frontend = Start-Process cmd -ArgumentList "/c npm run dev" -PassThru
$backend = Start-Process julia -ArgumentList "--project=. --compiled-modules=no student_io_core.jl" -PassThru

Write-Host "StudentIO System Running..."
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend: http://localhost:8080"
Write-Host "Press any key to stop..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Stop-Process -Id $frontend.Id
Stop-Process -Id $backend.Id
