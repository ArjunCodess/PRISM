$api = Start-Process -PassThru -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","apps.api.main:app","--app-dir","apps/api","--port","8000"
Start-Sleep -Seconds 2
Push-Location apps/web
if (-not (Test-Path "node_modules")) { npm install }
npm run dev
Pop-Location
Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
