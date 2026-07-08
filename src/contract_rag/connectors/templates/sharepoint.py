"""SharePoint / OneDrive connector TEMPLATE (S8 reusable asset) — INERT.

Auth: Microsoft Graph app registration (client-credentials flow); scope
`Sites.Read.All` / `Files.Read.All`. Acquire a token with `msal`.
API surface:
  - list: GET /v1.0/sites/{site-id}/drives/{drive-id}/root/children
  - fetch: GET /v1.0/.../items/{item-id}/content  -> write bytes to a temp file
Lazy-import `msal` and `requests` INSIDE the methods; never at module top level
(keeps the credential-free CI import clean). Do not auto-register — inject the
instance into the ingestion entry point.
"""
from __future__ import annotations

from contract_rag.connectors.templates._base import TemplateConnector


class SharePointConnector(TemplateConnector):
    name = "sharepoint"
