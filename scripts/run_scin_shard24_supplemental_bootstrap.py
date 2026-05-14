from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.case_loader import load_case_by_index

PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")
RUN_ID = "scin_hulumed_agent_70train_30test_20260513T190318Z"
RUN_ROOT = PROJECT_ROOT / "paper_data/final_30_70_large_runs" / RUN_ID / "machine_0"
COMBO_ROOT = RUN_ROOT / "hulumed/scin"
SHARD_ROOT = COMBO_ROOT / "bootstrap_shards"
ASSET_ROOT = RUN_ROOT / "assets" / f"{RUN_ID}_hulumed_scin"
POLICY_ROOT = ASSET_ROOT / "policy"
DATA_ROOT = PROJECT_ROOT / "data/scin"
LOG_ROOT = RUN_ROOT / "logs/supplemental_shard24"
MODEL_NAME = "Hulu-Med-7B"


def log(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}"
    print(line, flush=True)


def wait_ready(port: int, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer EMPTY"})
            with urlopen(req, timeout=5) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"Hulu-Med service on port {port} did not become ready")


def start_service(gpu: int, port: int) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pid_path = RUN_ROOT / "pids" / f"supp_hulumed_gpu{gpu}_{port}.pid"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PORT": str(port),
            "HOST": "127.0.0.1",
            "OPENAI_API_KEY": "EMPTY",
            "FORCE_RESTART": "1",
            "MAX_NUM_SEQS": "1",
            "MAX_MODEL_LEN": "8192",
            "GPU_MEMORY_UTILIZATION": "0.90",
            "LOG_FILE": str(LOG_ROOT / f"service_gpu{gpu}_{port}.log"),
            "PID_FILE": str(pid_path),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    with (LOG_ROOT / f"service_gpu{gpu}_{port}.startup.log").open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            ["bash", str(PROJECT_ROOT / "scripts/start_hulumed_server.sh"), str(PROJECT_ROOT)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    wait_ready(port)
    return pid_path


def stop_service(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    text = pid_path.read_text(encoding="utf-8", errors="ignore").strip()
    if text.isdigit():
        pid = int(text)
        subprocess.run(["bash", "-lc", f"pkill -TERM -P {pid} 2>/dev/null || true; kill -TERM {pid} 2>/dev/null || true"], check=False)
        time.sleep(5)
        subprocess.run(["bash", "-lc", f"pkill -KILL -P {pid} 2>/dev/null || true; kill -KILL {pid} 2>/dev/null || true"], check=False)
    pid_path.unlink(missing_ok=True)


def completed_case_ids() -> set[str]:
    return {path.parent.name for path in SHARD_ROOT.glob("shard_*/records/*/case_execution_record.json")}


def shard24_remaining_indices() -> list[int]:
    case_indices = json.loads((SHARD_ROOT / "shard_00024/case_indices.json").read_text(encoding="utf-8"))
    done_ids = completed_case_ids()
    remaining: list[int] = []
    for case_index in case_indices:
        case_input = load_case_by_index(int(case_index), DATA_ROOT)
        if str(case_input.case_id) not in done_ids:
            remaining.append(int(case_index))
    return remaining


def base_env(gpu: int, port: int) -> dict[str, str]:
    split_state_root = SHARD_ROOT / f"shard_00024_supp_gpu{gpu}" / "split_states"
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_MODEL": MODEL_NAME,
            "DERMAGENT_POLICY_ROOT": str(POLICY_ROOT),
            "DERMAGENT_SPLIT_STATE_ROOT": str(split_state_root),
            "DERMAGENT_DATA_ROOT": str(PROJECT_ROOT / "data"),
            "DERMAGENT_SCIN_LABEL_SPACE_ID": "scin_grouped",
            "OPENAI_TIMEOUT": "300",
        }
    )
    return env


def worker(gpu: int, port: int, tasks: queue.Queue[int], failures: list[dict[str, str]]) -> None:
    pid_path = start_service(gpu, port)
    shard_dir = SHARD_ROOT / f"shard_00024_supp_gpu{gpu}"
    output_dir = shard_dir / "records"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(gpu, port)
    try:
        while True:
            try:
                case_index = tasks.get_nowait()
            except queue.Empty:
                break
            cmd = [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/debug_single_case.py"),
                "--case-index",
                str(case_index),
                "--data-root",
                str(DATA_ROOT),
                "--output-dir",
                str(output_dir),
                "--enable-writeback",
                "--data-split",
                "train",
                "--run-mode",
                f"{RUN_ID}_hulumed_scin_bootstrap_supplemental",
                "--client-base-url",
                f"http://127.0.0.1:{port}/v1",
                "--client-api-key",
                "EMPTY",
                "--client-model",
                MODEL_NAME,
                "--client-timeout",
                "300",
                "--client-max-retries",
                "4",
            ]
            log_path = LOG_ROOT / f"gpu{gpu}_case_{case_index}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                failure = {"gpu": str(gpu), "case_index": str(case_index), "returncode": str(completed.returncode), "log": str(log_path)}
                failures.append(failure)
                log(f"FAILED supplemental case_index={case_index} gpu={gpu} rc={completed.returncode}")
            else:
                log(f"DONE supplemental case_index={case_index} gpu={gpu}")
            tasks.task_done()
    finally:
        (shard_dir / "DONE.json").write_text(
            json.dumps({"gpu": gpu, "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2) + "\n",
            encoding="utf-8",
        )
        stop_service(pid_path)


def main() -> int:
    remaining = shard24_remaining_indices()
    log(f"remaining shard24 cases: {len(remaining)}")
    if not remaining:
        return 0
    tasks: queue.Queue[int] = queue.Queue()
    for case_index in remaining:
        tasks.put(case_index)
    failures: list[dict[str, str]] = []
    threads = [
        threading.Thread(target=worker, args=(gpu, 8300 + gpu, tasks, failures), daemon=False)
        for gpu in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        (LOG_ROOT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        log(f"supplemental completed with {len(failures)} failed cases")
    else:
        log("supplemental completed without failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
