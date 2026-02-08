from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from jose.utils import base64url_decode

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

try:
    from .supabase_rest import rest_select, rest_insert, filter_eq
except ImportError:
    from supabase_rest import rest_select, rest_insert, filter_eq

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/supabase")


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None
    name: str
    subscription_plan_id: str
    access_token: str


_JWKS_CACHE: dict[str, Any] | None = None
_JWKS_CACHE_AT: float = 0.0
_JWKS_CACHE_TTL_S: float = 10 * 60
_JWKS_LOCK = asyncio.Lock()


def _auth_jwks_url() -> str | None:
    override = (os.getenv("SUPABASE_AUTH_JWKS_URL") or "").strip()
    if override:
        return override
    base = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/auth/v1/.well-known/jwks.json"


def _jwks_timeout_s() -> float:
    raw = (os.getenv("SUPABASE_AUTH_JWKS_TIMEOUT_S") or "").strip()
    if not raw:
        return 15.0
    try:
        v = float(raw)
        return max(1.0, min(v, 60.0))
    except Exception:
        return 15.0


async def _get_jwks() -> dict[str, Any]:
    global _JWKS_CACHE, _JWKS_CACHE_AT
    now = time.time()
    if _JWKS_CACHE and (now - _JWKS_CACHE_AT) < _JWKS_CACHE_TTL_S:
        return _JWKS_CACHE

    async with _JWKS_LOCK:
        now = time.time()
        if _JWKS_CACHE and (now - _JWKS_CACHE_AT) < _JWKS_CACHE_TTL_S:
            return _JWKS_CACHE

        url = _auth_jwks_url()
        if not url:
            raise RuntimeError("SUPABASE_AUTH_JWKS_URL missing (and SUPABASE_URL not set)")

        timeout_s = _jwks_timeout_s()
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    res = await client.get(url)
                    res.raise_for_status()
                    jwks = res.json()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.2)
                continue

        if last_exc is not None:
            raise last_exc

        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise RuntimeError("Invalid JWKS response")

        _JWKS_CACHE = jwks
        _JWKS_CACHE_AT = time.time()
        return jwks


def _jwk_to_pem(key: dict[str, Any]) -> str:
    kty = key.get("kty")
    if kty == "EC":
        crv = key.get("crv")
        if crv != "P-256":
            raise RuntimeError(f"Unsupported EC curve: {crv!r}")
        x_raw = key.get("x")
        y_raw = key.get("y")
        if not isinstance(x_raw, str) or not isinstance(y_raw, str):
            raise RuntimeError("Invalid EC JWK (missing x/y)")
        x = int.from_bytes(base64url_decode(x_raw.encode("utf-8")), "big")
        y = int.from_bytes(base64url_decode(y_raw.encode("utf-8")), "big")
        pub = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key(default_backend())
        return pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("utf-8")

    if kty == "RSA":
        n_raw = key.get("n")
        e_raw = key.get("e")
        if not isinstance(n_raw, str) or not isinstance(e_raw, str):
            raise RuntimeError("Invalid RSA JWK (missing n/e)")
        n = int.from_bytes(base64url_decode(n_raw.encode("utf-8")), "big")
        e = int.from_bytes(base64url_decode(e_raw.encode("utf-8")), "big")
        pub = rsa.RSAPublicNumbers(e, n).public_key(default_backend())
        return pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("utf-8")

    raise RuntimeError(f"Unsupported JWK kty: {kty!r}")


async def _decode_access_token(token: str) -> dict[str, Any]:
    header = jwt.get_unverified_header(token)
    alg = header.get("alg")

    if alg == "HS256":
        jwt_secret = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()
        if not jwt_secret:
            raise HTTPException(status_code=503, detail="Backend auth not configured (SUPABASE_JWT_SECRET missing)")
        return jwt.decode(token, jwt_secret, algorithms=["HS256"], options={"verify_aud": False})

    if alg in ("ES256", "RS256"):
        try:
            jwks = await _get_jwks()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Backend auth not configured (JWKS fetch failed): {exc}") from exc

        kid = header.get("kid")
        keys = jwks.get("keys") or []
        chosen: dict[str, Any] | None = None
        if isinstance(kid, str) and kid:
            for k in keys:
                if isinstance(k, dict) and k.get("kid") == kid:
                    chosen = k
                    break
        if chosen is None:
            # Best effort: if there's exactly one key, use it.
            only = [k for k in keys if isinstance(k, dict)]
            if len(only) == 1:
                chosen = only[0]

        if chosen is None:
            raise HTTPException(status_code=503, detail="Backend auth misconfigured (no matching JWKS key)")

        pem = _jwk_to_pem(chosen)
        return jwt.decode(token, pem, algorithms=[alg], options={"verify_aud": False})

    raise HTTPException(status_code=401, detail=f"Unsupported JWT alg: {alg!r}")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> CurrentUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = await _decode_access_token(token)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        email = payload.get("email") if isinstance(payload.get("email"), str) else None

        name = "Member"
        user_metadata: Any = payload.get("user_metadata")
        if isinstance(user_metadata, dict):
            raw = user_metadata.get("full_name") or user_metadata.get("name")
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
    except JWTError as exc:
        raise credentials_exception from exc

    subscription_plan_id = "standard"
    try:
        prof = await rest_select(
            table="profiles",
            select="id,email,full_name,subscription_plan_id",
            filters=[filter_eq("id", user_id)],
            single=True,
            auth="user",
            access_token=token,
        )
        if isinstance(prof, dict):
            subscription_plan_id = str(prof.get("subscription_plan_id") or "standard")
            email = str(prof.get("email") or email or "") or None
            full_name = prof.get("full_name")
            if isinstance(full_name, str) and full_name.strip():
                name = full_name.strip()
    except Exception:
        # If profile doesn't exist yet (or DB unreachable), continue with defaults.
        pass

    # Best-effort: if the profile row isn't created yet for some reason, create it.
    try:
        if subscription_plan_id == "standard":
            await rest_insert(
                table="profiles",
                rows=[
                    {
                        "id": user_id,
                        "email": email,
                        "full_name": name,
                        "subscription_plan_id": "standard",
                    }
                ],
                auth="admin",
            )
    except Exception:
        pass

    return CurrentUser(
        id=str(user_id),
        email=email,
        name=(name or "Member"),
        subscription_plan_id=(subscription_plan_id or "standard"),
        access_token=token,
    )
