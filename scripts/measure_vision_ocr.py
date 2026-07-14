"""Vision-OCR adoption measurement driver (spec: 2026-07-13-vision-ocr-design.md).

Runs fincritical + degrade(light/medium/fax/shred) + realscan for each candidate
model against an already-serving OpenAI-compatible VLM endpoint, with per-model
cache/raw dirs so paddle's caches stay untouched. Instance orchestration is
operational — see RUNBOOK below.

RUNBOOK (Lambda A100-40GB, same playbook as measure_gpu.py runs):
  1. Launch a GPU instance (e.g. Lambda API, gpu_1x_a100_sxm4) with your SSH key.
  2. Serve dots.ocr:
       pip install "vllm>=0.9" && \\
       vllm serve rednote-hilab/dots.ocr --trust-remote-code --port 8000
  3. Tunnel:  ssh -i <your-key>.pem -L 8000:localhost:8000 ubuntu@<ip> -N &
  4. Run:     uv run python scripts/measure_vision_ocr.py --model dots
  5. Swap serving to DeepSeek-OCR 2:
       vllm serve deepseek-ai/DeepSeek-OCR-2 --trust-remote-code --port 8000
     then: uv run python scripts/measure_vision_ocr.py --model dsocr
  6. TERMINATE THE INSTANCE.
  Fallback if vllm serve rejects a model: the model card's official docker image.
  Exact HF repo ids / flags: verify on the model cards at run time; they pin their
  own vllm versions.

Env prerequisites locally: REALSCAN_DIR + REALSCAN_GT_DIR (Tobacco800/GEDI),
FINCRITICAL_DIR or HF auto-download, golden_set/ + data/ for degrade.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = {
    # max_tokens: generous loop cap sized to each model's context window
    # (dots.ocr 131k ctx -> 8192; DeepSeek-OCR max_model_len 8192 total, so the
    # completion cap must leave room for ~2k vision+prompt input tokens -> 4096)
    "dots": {"vlm_model": "rednote-hilab/dots.ocr", "key": "dots", "max_tokens": "8192"},
    # OCR-2 substituted with v1 (measured 2026-07-13): DeepseekOCR2ForCausalLM is not
    # supported by any vLLM compatible with the rig's CUDA 12.8 driver (0.11.2 supports
    # DeepseekOCRForCausalLM; newer vLLMs need a newer driver), and OCR-2's remote code
    # breaks on the pinned transformers (LlamaFlashAttention2 removed).
    "dsocr": {"vlm_model": "deepseek-ai/DeepSeek-OCR", "key": "dsocr", "max_tokens": "4096"},
}
DEGRADE_LEVELS = ["light", "medium", "fax", "shred"]
OUT_ROOT = Path("measure_vision_ocr_results")
CACHE_ROOT = Path.home() / ".cache" / "contract-rag"
PAGES = {"fincritical": 85, "realscan": 100, "degrade": 15}  # sec/page denominators


def _run(module: str, extra_env: dict[str, str], out_file: Path, dry: bool) -> dict:
    env = {**os.environ, **extra_env}
    cmd = [sys.executable, "-m", module]
    print(f"\n=== {module} {extra_env.get('DEGRADE_LEVEL', '')} -> {out_file}")
    if dry:
        print("    (dry-run)", {k: v for k, v in extra_env.items()})
        return {"dry_run": True}
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    elapsed = round(time.time() - t0, 1)
    return {"elapsed_s": elapsed, "out": str(out_file)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    key = m["key"]
    out_dir = OUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "VLM_ENDPOINT": args.endpoint,
        "VLM_MODEL": m["vlm_model"],
        "VLM_RAW_DIR": str(out_dir / "raw"),
        "EXTRACT_BACKEND": "rule",
        # generous cap: real page markdown is 1-3k tokens; only repetition loops
        # on degraded/noisy pages ever reach it (measured 100k+ tokens uncapped)
        "VLM_MAX_TOKENS": m.get("max_tokens", "8192"),
    }
    timings: dict[str, dict] = {}

    fin_out = out_dir / "fincritical.json"
    timings["fincritical"] = _run(
        "contract_rag.eval.fincritical",
        {**base,
         "FINCRITICAL_CACHE": str(CACHE_ROOT / f"fincriticaled-run-{key}"),
         "FINCRITICAL_OUT": str(fin_out)},
        fin_out, args.dry_run,
    )

    for level in DEGRADE_LEVELS:
        d_out = out_dir / f"degrade_{level}.json"
        timings[f"degrade_{level}"] = _run(
            "contract_rag.eval.degrade",
            {**base,
             "DEGRADE_LEVEL": level,
             # match the documented paddle baseline slice (5 docs x first 3 pages)
             # so degraded F1/invented-ratio compare like-for-like
             "DEGRADE_SET_SIZE": "5",
             "DEGRADE_MAX_PAGES": "3",
             "DEGRADE_CACHE": str(CACHE_ROOT / f"degrade-{key}"),
             "DEGRADE_OUT": str(d_out),
             "VLM_RAW_DIR": str(out_dir / "raw" / f"degrade_{level}")},
            d_out, args.dry_run,
        )

    rs_out = out_dir / "realscan.json"
    timings["realscan"] = _run(
        "contract_rag.eval.realscan",
        {**base,
         "REALSCAN_CACHE": str(CACHE_ROOT / f"realscan-{key}"),
         "REALSCAN_OUT": str(rs_out)},
        rs_out, args.dry_run,
    )

    for name, t in timings.items():
        pages = PAGES.get(name.split("_")[0])
        if pages and "elapsed_s" in t:
            t["sec_per_page"] = round(t["elapsed_s"] / pages, 2)
    (out_dir / "summary.json").write_text(json.dumps(timings, indent=2))
    print(f"\nDone. Timings -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
