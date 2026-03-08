"""Client helpers for calling the optimiser API Gateway endpoint."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import requests
from requests_aws4auth import AWS4Auth


DEFAULT_TIMEOUT_SECONDS = 30


class OptimiserApiError(RuntimeError):
    """Raised when the optimiser API cannot be reached or returns an error."""


def _api_base_url() -> str:
    base_url = os.environ.get("OPTIMISER_API_URL", "").rstrip("/")
    if not base_url:
        raise OptimiserApiError("OPTIMISER_API_URL is not configured.")
    return base_url


def _build_auth():
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        return None

    region = os.environ.get("AWS_REGION", "eu-west-2")
    return AWS4Auth(
        access_key,
        secret_key,
        region,
        "execute-api",
        session_token=session_token,
    )


def _request(method: str, path: str, *, json_body=None, params=None) -> dict:
    url = f"{_api_base_url()}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            auth=_build_auth(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise OptimiserApiError(f"Optimiser API request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise OptimiserApiError(
            f"Optimiser API returned non-JSON response with status {response.status_code}."
        ) from exc

    if response.status_code >= 400:
        message = payload.get("error") or payload.get("message") or response.reason
        raise OptimiserApiError(f"Optimiser API error ({response.status_code}): {message}")

    return payload


def submit_job(payload: dict) -> dict:
    return _request("POST", "/optimise", json_body=payload)


def get_job_status(job_id: str) -> dict:
    return _request("GET", "/optimise-status", params={"job_id": job_id})


def get_job_result(job_id: str) -> dict:
    return _request("GET", "/optimise-result", params={"job_id": job_id})
