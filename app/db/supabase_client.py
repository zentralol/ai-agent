"""Shared Supabase async client factory.

The agent talks to Supabase with the service role key for controlled reads
(preferences) and conversation persistence. This factory centralizes client
construction so those features do not each reimplement it.
"""

from __future__ import annotations

import logging

from supabase import AsyncClient, AsyncClientOptions, acreate_client

from app.config import Settings

logger = logging.getLogger("zentra_agent.db.supabase_client")


async def create_supabase_client(settings: Settings) -> AsyncClient | None:
    """Create a service-role Supabase client, or None when unconfigured.

    Returns None (rather than raising) when the URL or service role key is
    missing so callers can degrade gracefully in local/dev environments.
    """

    supabase_url = settings.supabase_url
    service_role_key = settings.supabase_service_role_key
    if not supabase_url or not service_role_key:
        return None

    logger.info(
        "supabase_client_init timeout_seconds=%s",
        settings.supabase_timeout_seconds,
    )
    return await acreate_client(
        supabase_url=supabase_url,
        supabase_key=service_role_key,
        options=AsyncClientOptions(
            postgrest_client_timeout=settings.supabase_timeout_seconds
        ),
    )
