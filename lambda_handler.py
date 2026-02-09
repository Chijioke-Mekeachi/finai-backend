"""
AWS Lambda entrypoint for the FastAPI app.

Handler value depends on how you package the zip:
- If the zip root contains the `backend/` package: `backend.lambda_handler.handler`
- If the zip root *is* the backend folder: `lambda_handler.handler`
"""

try:
    from .main import app
except ImportError:  # pragma: no cover
    from main import app  # type: ignore

from mangum import Mangum

handler = Mangum(app, lifespan="off")
