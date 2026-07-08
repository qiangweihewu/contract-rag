from __future__ import annotations


def test_referenced_api_still_exists():
    # The playbook tells readers to use these — fail if the symbols drift away.
    from contract_rag.connectors.local import LocalFilesystemConnector  # noqa: F401
    from contract_rag.verticals.base import Vertical  # noqa: F401
    from contract_rag.verticals.registry import register_vertical  # noqa: F401

    for name in ("classify_clause", "permission_tags", "normalize_gold",
                 "canonicalize_value", "entities", "empty_facts"):
        assert hasattr(Vertical, name), f"Vertical lost method: {name}"
