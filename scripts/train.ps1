.\.venv\Scripts\python.exe ml/src/pipeline.py
Copy-Item ml/artifacts/demo_cases.json apps/web/public/demo_cases.json -Force
Copy-Item ml/artifacts/metrics.json apps/web/public/metrics.json -Force
Copy-Item ml/artifacts/model_card.json apps/web/public/model_card.json -Force
Write-Host "training complete"
