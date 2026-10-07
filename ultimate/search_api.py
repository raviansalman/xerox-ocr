#!/usr/bin/env python3
"""
Search-only ASGI entrypoint.

This module is used by the search-server compose stack:
  uvicorn search_api:app
"""

from ultimate_ui import create_fastapi_app


app = create_fastapi_app()

