"""
Vercel Serverless Function entrypoint.

With `backend/` set as the Vercel project root, this function handles all routes
via `backend/vercel.json`.
"""

from main import app
