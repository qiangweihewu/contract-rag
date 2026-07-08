# tests/test_abac.py
from contract_rag.security.abac import Principal, allowed_tags_for


def test_viewer_only_sees_general():
    assert allowed_tags_for(Principal(subject="u1", roles=["viewer"])) == ["general"]


def test_legal_role_sees_legal_family_and_restricted():
    tags = set(allowed_tags_for(Principal(subject="u2", roles=["legal"])))
    assert {"legal", "legal:ip", "restricted", "general"} <= tags
    assert "finance" not in tags


def test_multiple_roles_union_and_sorted():
    tags = allowed_tags_for(Principal(subject="u3", roles=["finance", "viewer"]))
    assert tags == sorted(set(tags))
    assert "finance" in tags and "general" in tags


def test_unknown_role_contributes_nothing():
    assert allowed_tags_for(Principal(subject="u4", roles=["martian"])) == []
