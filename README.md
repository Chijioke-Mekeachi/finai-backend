# FinTrack Pro API (FastAPI)

This backend is designed to be **stateless** and **serverless-friendly**:
- No local database/filesystem state (data lives in Supabase).
- No background workers or websockets required.

## Environment

Copy `backend/.env.example` to `backend/.env` for local development only.

Required at runtime:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- Plus optional values for AI / Paystack (see `backend/.env.example`)

## Local development

Install dev deps:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run (loads `backend/.env` only when `LOAD_DOTENV=1`):

```bash
LOAD_DOTENV=1 uvicorn main:app --reload --port 8000
```

## AWS Lambda

Use the handler:
- If your zip root contains the `backend/` package: `backend.lambda_handler.handler`
- If your zip root *is* the backend folder: `lambda_handler.handler`

Runtime deps are in `backend/requirements.txt` (no `uvicorn`).
