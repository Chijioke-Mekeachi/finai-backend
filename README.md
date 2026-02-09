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

Run (auto-loads `backend/.env` locally; set `LOAD_DOTENV=0` to disable):

```bash
uvicorn main:app --reload --port 8000
```

## AWS Lambda

Use the handler:
- If your zip root contains the `backend/` package: `backend.lambda_handler.handler`
- If your zip root *is* the backend folder: `lambda_handler.handler`

Runtime deps are in `backend/requirements.txt` (no `uvicorn`).

## Vercel

- Set the Vercel Project Root Directory to `backend/`.
- The serverless entrypoint is `api/index.py` (exports `app`).
- Routing is configured by `backend/vercel.json`.
