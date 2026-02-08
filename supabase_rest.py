from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _base_url() -> str:
    return _required_env("SUPABASE_URL").rstrip("/")


def _anon_key() -> str:
    return _required_env("SUPABASE_ANON_KEY")


def _service_key() -> str:
    return _required_env("SUPABASE_SERVICE_ROLE_KEY")


def _rest_url(path: str) -> str:
    return f"{_base_url()}{path}"


def _admin_headers() -> dict[str, str]:
    key = _service_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _user_headers(access_token: str) -> dict[str, str]:
    key = _anon_key()
    token = (access_token or "").strip()
    if not token:
        raise RuntimeError("user access token is required")
    return {
        "apikey": key,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def filter_eq(field: str, value: str) -> tuple[str, str]:
    return field, f"eq.{value}"


def filter_neq(field: str, value: str) -> tuple[str, str]:
    return field, f"neq.{value}"


def filter_gte(field: str, value: str) -> tuple[str, str]:
    return field, f"gte.{value}"


def filter_lte(field: str, value: str) -> tuple[str, str]:
    return field, f"lte.{value}"


def filter_like(field: str, pattern: str) -> tuple[str, str]:
    return field, f"like.{pattern}"


def filter_in(field: str, values: list[str]) -> tuple[str, str]:
    safe = [v for v in values if v]
    return field, f"in.({','.join(safe)})"


@dataclass(frozen=True)
class CountResult:
    exact: int


async def rest_select(
    *,
    table: str,
    select: str = "*",
    filters: list[tuple[str, str]] | None = None,
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    single: bool = False,
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> Any:
    params: dict[str, str] = {"select": select}
    for k, v in (filters or []):
        params[k] = v
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)
    if offset is not None:
        params["offset"] = str(offset)

    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    if single:
        headers = {**headers, "Accept": "application/vnd.pgrst.object+json"}

    url = _rest_url(f"/rest/v1/{table}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(url, params=params, headers=headers)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase select failed ({res.status_code}): {res.text[:500]}")
    return res.json()


async def rest_insert(
    *,
    table: str,
    rows: list[dict[str, Any]],
    returning: bool = False,
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> Any:
    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    if returning:
        headers = {**headers, "Prefer": "return=representation"}
    url = _rest_url(f"/rest/v1/{table}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, headers=headers, json=rows)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase insert failed ({res.status_code}): {res.text[:500]}")
    return res.json() if returning else None


async def rest_update(
    *,
    table: str,
    patch: dict[str, Any],
    filters: list[tuple[str, str]],
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> None:
    params: dict[str, str] = {}
    for k, v in filters:
        params[k] = v
    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    url = _rest_url(f"/rest/v1/{table}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.patch(url, params=params, headers=headers, json=patch)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase update failed ({res.status_code}): {res.text[:500]}")


async def rest_delete(
    *,
    table: str,
    filters: list[tuple[str, str]],
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> None:
    params: dict[str, str] = {}
    for k, v in filters:
        params[k] = v
    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    url = _rest_url(f"/rest/v1/{table}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.delete(url, params=params, headers=headers)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase delete failed ({res.status_code}): {res.text[:500]}")


async def rest_count(
    *,
    table: str,
    filters: list[tuple[str, str]] | None = None,
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> CountResult:
    params: dict[str, str] = {"select": "id"}
    for k, v in (filters or []):
        params[k] = v
    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    headers = {**headers, "Prefer": "count=exact"}
    url = _rest_url(f"/rest/v1/{table}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.head(url, params=params, headers=headers)
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase count failed ({res.status_code}): {res.text[:500]}")
    content_range = res.headers.get("content-range") or ""
    # format: "0-0/123" or "*/123"
    total = content_range.split("/")[-1] if "/" in content_range else ""
    try:
        return CountResult(exact=int(total))
    except Exception:
        return CountResult(exact=0)


async def rest_rpc(
    *,
    fn: str,
    payload: dict[str, Any] | None = None,
    auth: Literal["admin", "user"] = "admin",
    access_token: str | None = None,
) -> Any:
    headers = _admin_headers() if auth == "admin" else _user_headers(access_token or "")
    url = _rest_url(f"/rest/v1/rpc/{fn}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, headers=headers, json=(payload or {}))
    if res.status_code >= 400:
        raise RuntimeError(f"Supabase rpc failed ({res.status_code}): {res.text[:500]}")
    return res.json()
