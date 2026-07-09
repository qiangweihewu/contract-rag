---
name: Bug report
about: Something doesn't work as expected
title: ""
labels: bug
---

**What happened**

A clear description of the bug.

**Expected behavior**

What you expected instead.

**Repro**

```bash
# minimal command that reproduces it, e.g.
EXTRACT_BACKEND=rule uv run python -m contract_rag.benchmark
```

**Environment**

- Python version: (`python --version`; note this repo pins 3.12)
- `uv` version:
- OS:
- Relevant env vars set (`EXTRACT_BACKEND`, `VERTICAL`, etc. — **do not paste API keys**):

**Additional context**

Logs, stack trace, or a minimal `DocumentIR`/fixture that triggers it, if applicable.
