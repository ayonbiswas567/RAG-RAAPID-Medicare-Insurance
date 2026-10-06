"""Run the eval set. Thin wrapper around ``raapid-rag eval`` so it can be launched as a script.

    .venv/Scripts/python eval/run_eval.py                    # full mode (needs OPENAI_API_KEY)
    .venv/Scripts/python eval/run_eval.py --mode retrieval   # no model, free, CI-friendly

Exit code is 0 when the pass rate meets --min-pass-rate (default 0.8), else 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raapid_rag.cli import app  # noqa: E402

if __name__ == "__main__":
    app(["eval", *sys.argv[1:]])
