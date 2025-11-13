# LightFM Flask Service

- Endpoints: `/health`, `/api/recommend`, `/api/update-behavior`, `/api/quiz`, `/api/retrain`
- Runs on: `http://localhost:5000` (configurable via `PYTHON_PORT`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # update values if needed
python app.py
```

## Environment Variables

**Required:**
- `MONGODB_URI` - MongoDB connection string (e.g., `mongodb://localhost:27017`)
- `MONGODB_DB` - Database name (e.g., `riseup`)

**Optional:**
- `PYTHON_PORT=5000` - Port to run Flask service
- `NEXT_PUBLIC_APP_URL=http://localhost:3000` - Frontend URL for CORS
- `NEXT_PUBLIC_APP_ALT_URL=http://localhost:3001` - Alternative frontend URL
- `LFM_BOOTSTRAP_FALLBACK=1` - Use bootstrap fallback if MongoDB unavailable

## How It Works

1. **On Startup**: Connects to MongoDB and loads:
   - All tasks from `tasks` collection
   - User-task interactions from `usertaskinteractions` collection
   - Builds interaction matrix and trains LightFM model

2. **Recommendations**: Uses trained model to predict task scores for users based on:
   - User's past interactions
   - Similar users' preferences (collaborative filtering)
   - Falls back to rule-based if insufficient data

3. **Learning**: Model automatically retrains when:
   - New interactions are added (picked up on next recommendation)
   - Manual retrain via `/api/retrain` endpoint

## Data Requirements

- **Minimum**: 10+ user-task interactions to train model
- **Optimal**: 100+ interactions from multiple users
- **Fallback**: Uses bootstrap model if insufficient data
