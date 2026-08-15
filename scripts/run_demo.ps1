$root = Split-Path -Parent $PSScriptRoot
if (-not $root) { $root = Get-Location }
Set-Location $root
$api = Start-Process -PassThru -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","main:app","--app-dir","apps/api","--host","127.0.0.1","--port","8000"
Start-Sleep -Seconds 2
Push-Location apps/web
if (-not (Test-Path "node_modules")) { npm install }
npm run dev
Pop-Location
Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
