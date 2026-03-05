# MotorSafe — Insurance-style demo

This folder is a **standalone web demo** that makes the Motor TPL Pricing Engine feel like a real insurance company quote flow. Use it on your portfolio site so visitors can get a “real” quote experience.

## What’s in here

- **index.html** — One-page “insurer” site: header, hero, quote form, result, and an “About this demo” section.
- **styles.css** — Professional insurance look (trustworthy, clean).
- **app.js** — Form → POST `/quote` → display premium and breakdown.

The demo sends the form data to the pricing API and shows **gross premium**, **decision** (BIND/REFER), and a short **breakdown** (λ, μ, pure premium).

## Run the demo locally

### 1. Start the pricing API (FastAPI)

From the **repository root** (so artifact paths resolve):

```bash
# From motor-tpl-pricing-engine/
uvicorn src.api.app:app --reload --port 8000
```

The API serves `POST /quote` and `GET /health`. CORS is enabled so the demo can call it from another origin.

### 2. Serve the demo page

**Option A — Same origin (simplest)**  
Use FastAPI to serve the static files so the demo and API are on the same origin and you don’t need to set `QUOTE_API_BASE`:

```bash
# From motor-tpl-pricing-engine/
uvicorn src.api.app:app --reload --port 8000
# Then mount static files (see "Serving the demo from FastAPI" below)
```

**Option B — Separate static server**  
Serve the `web-demo` folder with any static server and point the demo at the API:

```bash
# Terminal 1: API
uvicorn src.api.app:app --reload --port 8000

# Terminal 2: demo (e.g. from web-demo/)
npx serve -p 3000
# Open http://localhost:3000
```

Then set the API base URL before the form is used. In the browser console:

```js
window.QUOTE_API_BASE = 'http://localhost:8000';
```

Or add a small script in `index.html`:

```html
<script>window.QUOTE_API_BASE = 'http://localhost:8000';</script>
<script src="app.js"></script>
```

### 3. Get a quote

Fill the form and click **Get my quote**. You should see the annual premium (e.g. in GBP) and the short breakdown.

## Use it on your portfolio website

1. **Copy the demo**  
   Copy the contents of `web-demo/` (or the built assets if you use a build step) into your portfolio project, e.g. under a route like `/projects/motor-tpl-demo` or `/demo/motor-pricing`.

2. **Connect to your backend**  
   - If the portfolio and the pricing API are on the **same domain** (e.g. portfolio.com and api.portfolio.com with same-site cookies), set `window.QUOTE_API_BASE = 'https://your-pricing-api.com'` (or leave it empty if you proxy `/quote` to the API).
   - If they are on different domains, ensure the pricing API has CORS enabled (the FastAPI app in this repo already includes CORS for demo use). For production, restrict `allow_origins` to your portfolio domain.

3. **Deploy the API**  
   Deploy the Motor TPL engine (e.g. FastAPI on Railway, Render, or AWS Lambda + API Gateway) and point `QUOTE_API_BASE` to that URL.

4. **Keep the “demo” framing**  
   The page already states that it’s a portfolio demo and not a real insurer. You can change the branding (e.g. logo name “MotorSafe”) in `index.html` and `styles.css` to match your portfolio.

## Serving the demo from FastAPI (optional)

To serve the demo and the API from one app (one origin, no CORS needed for the demo), mount the `web-demo` directory and optionally add a route for `/` that returns `index.html`:

```python
# In src/api/app.py (optional)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# After creating app, mount static files (repo root = parents[2] from this file)
root = Path(__file__).resolve().parents[2]
app.mount("/demo", StaticFiles(directory=str(root / "web-demo"), html=True), name="demo")
```

Then open `http://localhost:8000/demo/` to use the demo; the form will POST to `/quote` on the same origin.

## Requirements

- Pricing API running (FastAPI with trained artifacts, or a Lambda + API Gateway that accepts `POST` body `{ "policy": { ... } }` and returns the same quote shape).
- Browser with JavaScript enabled.

No build step is required for the provided HTML/CSS/JS.
