"""Salesforce (Files / ContentVersion) connector TEMPLATE (S8 reusable asset) — INERT.

Auth: OAuth 2.0 connected app (JWT bearer or username-password); REST + SOQL.
API surface:
  - list: SOQL `SELECT Id, Title, FileExtension FROM ContentVersion WHERE IsLatest = true`
  - fetch: GET /services/data/vXX.0/sobjects/ContentVersion/{id}/VersionData
Lazy-import `simple_salesforce` (or `requests`) INSIDE the methods; never at module
top level. Do not auto-register — inject the instance.
"""
from __future__ import annotations

from contract_rag.connectors.templates._base import TemplateConnector


class SalesforceConnector(TemplateConnector):
    name = "salesforce"
