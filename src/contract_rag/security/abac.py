"""ABAC: map an identity (roles) to the permission tags it may retrieve.
Tags align with verticals/contract/enrich.py (general, finance, legal, legal:ip, restricted)."""
from __future__ import annotations

from pydantic import BaseModel


class Principal(BaseModel):
    subject: str
    roles: list[str]


ROLE_TAGS: dict[str, list[str]] = {
    "legal": ["legal", "legal:ip", "restricted", "general"],
    "finance": ["finance", "general"],
    "viewer": ["general"],
}


def allowed_tags_for(principal: Principal) -> list[str]:
    tags: set[str] = set()
    for role in principal.roles:
        tags.update(ROLE_TAGS.get(role, []))
    return sorted(tags)
