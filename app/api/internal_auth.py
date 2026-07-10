"""Inbound service-to-service authentication for the agent API.

The agent is an internal microservice reached only through the Express gateway.
Every gateway call carries a shared secret in the ``X-Internal-Service-Token``
header. This dependency verifies it in constant time.

When the secret is not configured (``AGENT_INTERNAL_TOKEN`` unset) inbound auth is
skipped and a warning is logged, so local development and CI work without a
secret while production stays protected once the token is set.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings

logger = logging.getLogger("zentra_agent.api.internal_auth")

INTERNAL_TOKEN_HEADER = "X-Internal-Service-Token"

# Module-level singletons so ``Depends``/``Header`` are not called in argument
# defaults (ruff B008), matching the pattern in ``app/api/agent.py``.
_TokenHeader = Header(default=None, alias=INTERNAL_TOKEN_HEADER)
_SettingsDependency = Depends(get_settings)


async def require_internal_auth(
    x_internal_service_token: str | None = _TokenHeader,
    settings: Settings = _SettingsDependency,
) -> None:
    """Reject calls that do not carry the configured internal service token."""

    expected = settings.agent_internal_token
    if not expected:
        logger.warning(
            "agent_internal_auth_unconfigured allowing request; set %s to enforce",
            "AGENT_INTERNAL_TOKEN",
        )
        return

    if x_internal_service_token is None or not hmac.compare_digest(
        x_internal_service_token, expected
    ):
        logger.warning(
            "agent_internal_auth_rejected header_present=%s",
            bool(x_internal_service_token),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal service token.",
        )
