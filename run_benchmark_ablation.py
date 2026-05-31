"""
ChatDev Ablation Benchmark Runner
Runs all 5 benchmarks WITHOUT Client Liaison — CEO receives prompt directly.

Usage:
    pip install httpx websockets
    python run_benchmarks_ablation.py

Results are saved to ./benchmark_results_ablation/<benchmark_name>_<session_id>.zip
"""

from __future__ import annotations

import asyncio
import json
import zipfile
import io
import httpx
import websockets
import re
import sys
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:6400"
WS_URL = "ws://localhost:6400/ws"
SCRIPT_DIR = Path(__file__).resolve().parent
YAML_PATH = SCRIPT_DIR / "yaml_instance" / "ChatDev_benchmarks_ablation.yaml"
OUTPUT_DIR = Path("benchmark_results_ablation")
POLL_INTERVAL = 15
MAX_POLL_TIME = 60 * 30

BENCHMARKS = {
    "benchmark_task_api":       "I need a task management REST API built.",
    "benchmark_csv_pipeline":   "I need a CSV processing pipeline built.",
    "benchmark_chat_server":    "I need a real-time chat server built.",
    "benchmark_url_shortener":  "I need a URL shortener service built.",
    "benchmark_expense_tracker":"I need an expense tracker CLI built.",
}

BENCHMARK_SPEC_PATTERN = re.compile(r"(BENCHMARK_SPEC:\s*)\$\{[^}]+\}")


def load_yaml_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def patch_yaml(template: str, benchmark_name: str) -> str:
    patched = BENCHMARK_SPEC_PATTERN.sub(
        rf"\g<1>${{{benchmark_name}}}",
        template,
    )
    if f"${{{benchmark_name}}}" not in patched:
        raise ValueError(f"Could not patch BENCHMARK_SPEC for {benchmark_name}")
    return patched


async def upload_yaml(client: httpx.AsyncClient, content: str, filename: str) -> None:
    put_resp = await client.put(
        f"{BASE_URL}/api/workflows/{filename}/update",
        json={"content": content},
        timeout=30,
    )
    if put_resp.status_code == 404:
        post_resp = await client.post(
            f"{BASE_URL}/api/workflows/upload/content",
            json={"filename": filename, "content": content},
            timeout=30,
        )
        post_resp.raise_for_status()
    else:
        put_resp.raise_for_status()


async def execute_workflow(
    client: httpx.AsyncClient, yaml_file: str, session_id: str, task_prompt: str
) -> str:
    resp = await client.post(
        f"{BASE_URL}/api/workflow/execute",
        json={
            "yaml_file": yaml_file,
            "task_prompt": task_prompt,
            "session_id": session_id,
            "attachments": [],
            "log_level": "INFO",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  [execute error] {resp.text}")
    resp.raise_for_status()
    returned = resp.json()
    return returned if isinstance(returned, str) else session_id


async def poll_and_download(
    client: httpx.AsyncClient,
    session_id: str,
    benchmark_name: str,
    output_dir: Path,
    done_event: asyncio.Event,
) -> Path | None:
    deadline = asyncio.get_event_loop().time() + MAX_POLL_TIME
    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await client.get(
                f"{BASE_URL}/api/sessions/{session_id}/download",
                timeout=60,
            )
            if resp.status_code == 200 and len(resp.content) > 5000:
                try:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        if any("execution_logs.json" in n for n in zf.namelist()):
                            out_path = output_dir / f"{benchmark_name}_{session_id}.zip"
                            out_path.write_bytes(resp.content)
                            done_event.set()
                            return out_path
                        else:
                            print(f"[{benchmark_name}] Still running (no execution_logs yet)...")
                except zipfile.BadZipFile:
                    pass
            elif resp.status_code == 500:
                print(f"[{benchmark_name}] Server error on download — stopping poll")
                break
        except (httpx.HTTPStatusError, httpx.RequestError):
            pass
        await asyncio.sleep(POLL_INTERVAL)
    done_event.set()
    return None


async def ws_connect_and_get_session() -> tuple[str, websockets.WebSocketClientProtocol]:
    ws = await websockets.connect(WS_URL, open_timeout=10)
    raw = await asyncio.wait_for(ws.recv(), timeout=10)
    data = json.loads(raw)
    session_id = data["data"]["session_id"]
    return session_id, ws


async def ws_keep_alive(
    ws: websockets.WebSocketClientProtocol,
    session_id: str,
    done_event: asyncio.Event,
) -> None:
    while not done_event.is_set():
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError:
            pass
        except websockets.ConnectionClosed:
            break
    try:
        await ws.close()
    except Exception:
        pass


async def run_benchmark(
    benchmark_name: str,
    task_prompt: str,
    yaml_template: str,
    output_dir: Path,
) -> dict:
    result = {
        "benchmark": benchmark_name,
        "session_id": None,
        "status": "unknown",
        "output_file": None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    }

    yaml_filename = f"ablation_{benchmark_name}.yaml"
    done_event = asyncio.Event()

    async with httpx.AsyncClient() as client:
        try:
            patched = patch_yaml(yaml_template, benchmark_name)
            print(f"[{benchmark_name}] Uploading YAML...")
            await upload_yaml(client, patched, yaml_filename)

            print(f"[{benchmark_name}] Connecting WebSocket...")
            session_id, ws = await ws_connect_and_get_session()
            result["session_id"] = session_id
            print(f"[{benchmark_name}] Session: {session_id[:8]}...")

            ws_task = asyncio.create_task(
                ws_keep_alive(ws, session_id, done_event)
            )

            print(f"[{benchmark_name}] Starting execution...")
            await execute_workflow(client, yaml_filename, session_id, task_prompt)
            print(f"[{benchmark_name}] Running — polling every {POLL_INTERVAL}s...")

            out_path = await poll_and_download(
                client, session_id, benchmark_name, output_dir, done_event
            )

            await ws_task

            result["finished_at"] = datetime.now().isoformat()
            if out_path:
                result["status"] = "success"
                result["output_file"] = str(out_path)
                print(f"[{benchmark_name}] ✓ Done → {out_path.name}")
            else:
                result["status"] = "timeout"
                print(f"[{benchmark_name}] ✗ Timed out after {MAX_POLL_TIME // 60} minutes")

        except Exception as e:
            done_event.set()
            result["status"] = "error"
            result["finished_at"] = datetime.now().isoformat()
            print(f"[{benchmark_name}] ✗ Error: {e}")

    return result


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        yaml_template = load_yaml_template(YAML_PATH)
    except FileNotFoundError:
        print(f"ERROR: YAML file not found at:\n  {YAML_PATH}")
        sys.exit(1)

    print(f"Starting {len(BENCHMARKS)} ablation benchmarks in parallel...\n")
    start = datetime.now()

    tasks = [
        run_benchmark(name, prompt, yaml_template, OUTPUT_DIR)
        for name, prompt in BENCHMARKS.items()
    ]
    results = await asyncio.gather(*tasks)

    elapsed = (datetime.now() - start).seconds // 60
    print(f"\n{'─' * 60}")
    print(f"All ablation benchmarks finished in ~{elapsed} minutes\n")

    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['benchmark']:<30} {r['status']}")
        if r["output_file"]:
            print(f"      → {r['output_file']}")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())