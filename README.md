# PRISM

Predictive Risk Intelligence for Space Monitoring is an explainable T-48-hour conjunction-risk copilot. It is an educational research prototype, not flight software.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python ml/src/pipeline.py
cd apps/web
npm install
npm run build
```

Run the demo:

```powershell
.\scripts\run_demo.ps1
```

- UI: http://localhost:3000
- API: http://localhost:8000/docs

The UI also works offline from cached `public/demo_cases.json` if the API is stopped.

## Layout

- `ml/` training, evaluation, SHAP, artifacts
- `apps/api` FastAPI inference
- `apps/web` Next.js mission-control UI
- `prd.md` locked product requirements

## Safety

Forecasts are advisory. Human approval is required. Do not use PRISM for operational collision-avoidance decisions.
