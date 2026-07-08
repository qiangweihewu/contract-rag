"""Confluence connector TEMPLATE (S8 reusable asset) — INERT.

Auth: Atlassian API token (Basic auth: email + token) for Confluence Cloud REST.
API surface:
  - list: GET /wiki/rest/api/content?type=page&expand=body.storage (paginated)
  - fetch: GET /wiki/rest/api/content/{id}?expand=body.storage  -> export the body
Lazy-import `requests` INSIDE the methods; never at module top level.
Do not auto-register — inject the instance.
"""
from __future__ import annotations

from contract_rag.connectors.templates._base import TemplateConnector


class ConfluenceConnector(TemplateConnector):
    name = "confluence"
