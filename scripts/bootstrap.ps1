python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
if (-not (Test-Path "apps/web/node_modules")) {
  Push-Location apps/web
  npm install
  Pop-Location
}
Write-Host "prism bootstrap complete"
