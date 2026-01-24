# StudentIO Frontend Deployment Script
# Builds and deploys frontend to GitHub Pages

Write-Host "🚀 Deploying StudentIO Frontend to GitHub Pages..." -ForegroundColor Cyan
Write-Host ""

# Build the frontend
Write-Host "📦 Building production bundle..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Build successful!" -ForegroundColor Green
Write-Host ""

# Deploy to GitHub Pages
Write-Host "📤 Deploying to gh-pages branch..." -ForegroundColor Yellow

# Create .nojekyll in dist
New-Item -ItemType File -Path "dist\.nojekyll" -Force | Out-Null

# Deploy using git subtree
git add dist -f
git commit -m "Build for deployment" --allow-empty
git subtree push --prefix dist origin gh-pages

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✨ Deployment successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your site will be available at:" -ForegroundColor Cyan
    Write-Host "https://YOUR-USERNAME.github.io/StudentIO-v2/" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host ""
    Write-Host "Note: It may take a few minutes for changes to appear" -ForegroundColor Gray
}
else {
    Write-Host ""
    Write-Host "✗ Deployment failed!" -ForegroundColor Red
    Write-Host "Make sure you have committed your changes and have write access to the repository." -ForegroundColor Yellow
}
