---
name: Feature request
about: Suggest a new capability, cleaner, backend, or vertical
title: ""
labels: enhancement
---

**Problem**

What's missing or awkward today, and why it matters.

**Proposed change**

What you'd add or change. If it's a new cleaner, note the matching dirtifier it should pair
with (see `CLAUDE.md` — "`dirtify` mirrors `clean`"). If it's a new extraction backend or
vertical, note which existing seam it should plug into (`Extractor` protocol, `verticals/`
package shape).

**How would this be measured**

This project's numbers all come from reproducible evals — if the feature is meant to move a
metric, say which one (field-F1, source-attribution accuracy, Context Recall, quality score)
and against which dataset/golden set.

**Alternatives considered**

Anything you ruled out and why (a negative result is useful context here too).
