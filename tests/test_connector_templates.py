from __future__ import annotations

import pytest

from contract_rag.connectors.base import Connector, SourceRef
from contract_rag.connectors.templates.confluence import ConfluenceConnector
from contract_rag.connectors.templates.gdrive import GoogleDriveConnector
from contract_rag.connectors.templates.salesforce import SalesforceConnector
from contract_rag.connectors.templates.sharepoint import SharePointConnector

_ALL = [SharePointConnector, GoogleDriveConnector, SalesforceConnector, ConfluenceConnector]


@pytest.mark.parametrize("cls", _ALL)
def test_template_is_structurally_a_connector(cls):
    conn = cls(config_key="value")
    assert isinstance(conn, Connector)
    assert isinstance(conn.name, str) and conn.name


@pytest.mark.parametrize("cls", _ALL)
def test_template_operations_are_inert(cls):
    conn = cls()
    with pytest.raises(NotImplementedError):
        list(conn.list_documents())
    with pytest.raises(NotImplementedError):
        conn.fetch(SourceRef(id="x", name="x"))


def test_distinct_names():
    assert {c().name for c in _ALL} == {"sharepoint", "gdrive", "salesforce", "confluence"}
