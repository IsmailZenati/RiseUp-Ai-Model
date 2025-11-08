# LightFM Flask Service

- Endpoints: `/health`, `/api/recommend`, `/api/update-behavior`
- Runs on: `http://localhost:5000` (configurable via `PYTHON_PORT`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env  # update values if needed
python app.py
```

## Env

- `PYTHON_PORT=5000`
- `NEXT_PUBLIC_APP_URL=http://localhost:3000`
- `NEXT_PUBLIC_APP_ALT_URL=http://localhost:3001`
- `MONGODB_URI=` (optional for future data-driven training)
- `MONGODB_DB=` (optional)
- `LFM_BOOTSTRAP_FALLBACK=1` (keeps small bootstrap catalog for recommendations)
