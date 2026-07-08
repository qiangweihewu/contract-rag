"""Back-compat shim. The contract facts schema now lives in
contract_rag.verticals.contract.schema. Re-exported so existing imports keep working."""
from __future__ import annotations

from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts

__all__ = ["ContractFacts", "ExtractedClause"]
