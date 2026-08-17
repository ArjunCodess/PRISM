# Deploy the PRISM exhibit and API

PRISM is two services. The website talks **only** to the live API. There is no frozen-JSON fallback in development or production. If FastAPI is down, the site shows an error.

- **Website** (`apps/web`): Next.js exhibit.
- **API** (`apps/api`): FastAPI. Serves cases, metrics, and live `POST /v1/risk/predict`.

Copy `.env.example` to `.env.local` (website) or set the same names on the host. See `apps/web/.env.example`. `NEXT_PUBLIC_API_URL` is required.

The intended exhibit is a **local laptop prototype**: Next + FastAPI on the presentation machine, Wi-Fi off if you want, API still running. A public Vercel deploy without an always-on API is a worse exhibit.

There is no public host in this repository. After you deploy, put the website URL in the README.

## What you need

- Node.js 20+
- Python 3.11+
- Frozen artifacts already in `ml/artifacts/` (`demo_cases.json`, `metrics.json`, `model_card.json`, `feature_schema.json`, `risk_regressor.json`, `warning_calibrator.joblib`)
- A production build of the web app (`python main.py --build-only` or `npm run build` in `apps/web`)
- `NEXT_PUBLIC_API_URL` pointing at FastAPI (local: `http://127.0.0.1:8000`)

Do not deploy `data/raw/`. The ESA archive is large and is not required at inference time.

## 1. Website (Vercel, optional)

The public site needs the API. Set this **before** you build:

| Name | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | `https://your-api-host.example` with no trailing slash |

`NEXT_PUBLIC_*` is inlined at **build** time. After you add or change it, redeploy the website.

1. Install the Vercel CLI and log in: `npm i -g vercel` then `vercel login`.
2. From the repository root:

```powershell
cd apps/web
copy .env.example .env.local
# edit .env.local so NEXT_PUBLIC_API_URL is your live API
vercel
```

3. In the Vercel project settings, set:
   - **Root Directory:** `apps/web`
   - **Framework Preset:** Next.js
   - **Build Command:** `npm run build`
   - **Install Command:** `npm ci`
   - **Node version:** 20.x
   - **Environment variable:** `NEXT_PUBLIC_API_URL`

4. Production deploy:

```powershell
cd apps/web
vercel --prod
```

Git integration also works: import the GitHub repo in Vercel, set the root directory to `apps/web`, and production deploys from `main`. Add `NEXT_PUBLIC_API_URL` in Vercel env settings for Production.

Each case page calls `POST /v1/risk/predict` so the forecast on screen is a live model run. The lab and case list load from `GET /v1/cases` and `GET /v1/model-card`. If the API is down, the site errors instead of showing a silent cache.

## 2. API (any Python host)

The API process must see the repository layout, because it loads models from `ml/artifacts/` and source from `ml/src/`. Case JSON lives in `ml/artifacts/demo_cases.json` as the API store, not as a Next fallback.

Minimum files on the host:

```
apps/api/main.py
ml/src/          # inference, features, constants, ...
ml/artifacts/    # trained models and frozen JSON
requirements.txt
```

Start command from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "ml/src"
python -m uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=ml/src
python -m uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port ${PORT:-8000}
```

Health check: `GET /health` should return `{"status":"ok",...}`. Docs: `/docs`.

CORS is already open (`allow_origins=["*"]`), so a Vercel frontend can call this API.

### Typical hosts

**Render / Railway / Fly.io**

- Runtime: Python 3.11
- Build: `pip install -r requirements.txt`
- Start: `PYTHONPATH=ml/src python -m uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port $PORT`
- Include `ml/artifacts/` in the deploy (do not gitignore the joblib/json models you need)
- Memory: 512 MB is usually enough; SHAP loads with the model on first predict

**A VPS**

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=ml/src
python -m uvicorn main:app --app-dir apps/api --host 127.0.0.1 --port 8000
```

Put Caddy or nginx in front for HTTPS, and proxy `/` to `127.0.0.1:8000`.

Do not use `npx convex deploy` for this project. There is no Convex backend.

## 3. Point the website at the API

1. Deploy the API and copy its HTTPS origin, for example `https://prism-api.onrender.com`.
2. Set `NEXT_PUBLIC_API_URL` on the website project to that origin.
3. Redeploy the website.
4. Confirm:
   - exhibit home still lists six cases
   - `https://your-api/health` is ok
   - stopping the API shows the Next error page, not cached cases
   - case reveal still works (reveal uses the Next route, not the Python API)
   - live predict hits `POST /v1/risk/predict`

## 4. Local production-like check

```powershell
python main.py --build-only
python main.py --skip-download --skip-train --skip-graphs --skip-checks
```

That serves the production Next build at `http://127.0.0.1:3000` and the API at `http://127.0.0.1:8000`. Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` before building.

To run them separately:

```powershell
$env:PYTHONPATH = "ml/src"
python -m uvicorn main:app --app-dir apps/api --host 127.0.0.1 --port 8000
```

```powershell
cd apps/web
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run start -- --hostname 127.0.0.1 --port 3000
```

## 5. After it is live

Add the public website URL to the README **Exhibit** and **Running** sections so a reader does not have to clone the repo to see the six cases.

Keep the disclaimer on the deployed site: research prototype, not flight software, not an operational decision system.
