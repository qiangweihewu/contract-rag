"""Google Drive connector TEMPLATE (S8 reusable asset) — INERT.

Auth: a service account with domain-wide delegation, or OAuth user creds; scope
`https://www.googleapis.com/auth/drive.readonly`.
API surface:
  - list: files().list(q="mimeType != 'application/vnd.google-apps.folder'")
  - fetch: files().get_media(fileId=...)  -> write bytes to a temp file
Lazy-import `google.oauth2` / `googleapiclient` INSIDE the methods; never at module
top level. Do not auto-register — inject the instance.
"""
from __future__ import annotations

from contract_rag.connectors.templates._base import TemplateConnector


class GoogleDriveConnector(TemplateConnector):
    name = "gdrive"
