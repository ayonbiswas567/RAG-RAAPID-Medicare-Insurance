"""Process-level smoke test for the two surfaces.

Starts the FastAPI app under uvicorn and the Streamlit app as real subprocesses (real embedding
model, real language model), waits for their health endpoints, exercises the API over HTTP, and
writes the responses to docs/runs/step7_api_smoke.json.  Then terminates both.

Run:  .venv/Scripts/python eval/step7_smoke.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API_PORT, UI_PORT = 8765, 8599
API = f"http://127.0.0.1:{API_PORT}"
UI = f"http://127.0.0.1:{UI_PORT}"


def wait_for(url: str, timeout: float, expect_text: str | None = None) -> float:
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200 and (expect_text is None or expect_text in r.text):
                return round(time.perf_counter() - t0, 1)
            last = f"{r.status_code} {r.text[:80]}"
        except httpx.HTTPError as exc:
            last = str(exc)[:80]
        time.sleep(1)
    raise SystemExit(f"{url} not ready after {timeout}s (last: {last})")


def main() -> None:
    # Server logs go to the temp dir, not the repo: Streamlit prints the machine's network and
    # external URLs at startup, which do not belong in a shared document folder.
    log_dir = Path(tempfile.gettempdir())
    api_log = open(log_dir / "raapid_step7_uvicorn.log", "w", encoding="utf-8")
    ui_log = open(log_dir / "raapid_step7_streamlit.log", "w", encoding="utf-8")
    print(f"server logs: {log_dir}")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "raapid_rag.api:app", "--port", str(API_PORT)], stdout=api_log, stderr=subprocess.STDOUT, cwd=ROOT)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "raapid_rag" / "ui" / "app.py"),
                           "--server.headless", "true", "--server.port", str(UI_PORT)], stdout=ui_log, stderr=subprocess.STDOUT, cwd=ROOT)
    results: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        results["api_ready_seconds"] = wait_for(f"{API}/health", timeout=180)
        results["ui_ready_seconds"] = wait_for(f"{UI}/_stcore/health", timeout=180, expect_text="ok")
        print(f"api ready in {results['api_ready_seconds']}s, streamlit ready in {results['ui_ready_seconds']}s")

        with httpx.Client(base_url=API, timeout=120) as c:
            results["health"] = c.get("/health").json()
            results["documents"] = c.get("/documents").json()
            r = c.post("/search", json={"query": "audio-only telehealth encounters"})
            results["search"] = {"status_code": r.status_code, "passes_gate": r.json()["passes_gate"], "chunk_ids": [x["chunk_id"] for x in r.json()["chunks"]]}
            calls = [
                ("query_hcc", {"query": "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?"}),
                ("query_gate_refusal", {"query": "Who won the 2024 World Series?"}),
                ("query_scoped_no_trace", {"query": "What does section 2.2 say about audio-only encounters?", "doc_ids": ["policy_002"], "include_trace": False}),
                ("query_blank_400", {"query": "   "}),
            ]
            for label, body in calls:
                t0 = time.perf_counter()
                r = c.post("/query", json=body)
                elapsed = round((time.perf_counter() - t0) * 1000)
                entry = {"status_code": r.status_code, "request_id": r.headers.get("X-Request-ID"), "wall_ms": elapsed}
                if r.status_code == 200:
                    a = r.json()
                    entry.update({"answer_status": a["answer_status"], "confidence": a["confidence"], "band": a["confidence_band"],
                                  "needs_human_review": a["needs_human_review"], "citations": [(x["chunk_id"], x["verified"]) for x in a["citations"]],
                                  "tools": [t["name"] for t in a["tool_calls"]], "trace_keys": sorted(a["trace"]), "answer": a["answer"][:200]})
                else:
                    entry["detail"] = r.json().get("detail")
                results[label] = entry
                print(f"[{label}] {entry['status_code']} {entry.get('answer_status', entry.get('detail'))} "
                      f"conf={entry.get('confidence')} review={entry.get('needs_human_review')} tools={entry.get('tools')} {elapsed} ms id={entry['request_id']}")
        results["streamlit_health"] = httpx.get(f"{UI}/_stcore/health", timeout=5).text
    finally:
        api.terminate()
        ui.terminate()
        api.wait(timeout=30)
        ui.wait(timeout=30)
        api_log.close()
        ui_log.close()

    out = ROOT / "docs" / "runs" / "step7_api_smoke.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
