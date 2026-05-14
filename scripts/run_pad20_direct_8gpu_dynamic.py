from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

from agent.evaluation_protocol import EvaluationTargetSpec, run_evaluation_suite
from agent.experiment_state import load_split_payload
from agent.policy_config import load_stable_policy
from agent.policy_evaluation import build_policy_summary
from integrations.openai_client import DermOpenAIClient


MODELS: dict[str, dict[str, str]] = {
    "qwen": {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "start_script": "scripts/start_qwen_server.sh",
        "max_model_len": "16384",
        "gpu_memory_utilization": "0.82",
    },
    "llama": {
        "model_name": "Llama-3.2-11B-Vision-Instruct",
        "start_script": "scripts/start_llama_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
    "skinvl": {
        "model_name": "SkinVL-MM",
        "start_script": "scripts/start_skinvl_server.sh",
        "max_model_len": "",
        "gpu_memory_utilization": "",
    },
    "hulumed": {
        "model_name": "Hulu-Med-7B",
        "start_script": "scripts/start_hulumed_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
    "medgemma": {
        "model_name": "medgemma-4b-it",
        "start_script": "scripts/start_medgemma_server.sh",
        "max_model_len": "12288",
        "gpu_memory_utilization": "0.85",
    },
    "dermatollama": {
        "model_name": "DermatoLlama-full",
        "start_script": "scripts/start_dermatollama_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
}


@dataclass(frozen=True)
class ShardTask:
    model: str
    shard_id: int
    offset: int
    limit: int


class Pad20DirectRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = args.output_root.resolve()
        self.logs_root = self.run_root / "logs"
        self.pids_root = self.run_root / "pids"
        self.shards_root = self.run_root / "shards"
        self.reports_root = self.run_root / "reports"
        for path in (self.run_root, self.logs_root, self.pids_root, self.shards_root, self.reports_root):
            path.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.failures: list[dict[str, Any]] = []
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, _signum: int, _frame: Any) -> None:
        self.stop_event.set()

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}"
        with self.lock:
            print(line, flush=True)
            with (self.run_root / "runner.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def gpu_ids(self) -> list[int]:
        values = [int(item) for item in str(self.args.gpus).split(",") if item.strip()]
        if not values:
            raise ValueError("No GPU ids were provided.")
        return values

    def model_keys(self) -> list[str]:
        values = [item.strip() for item in str(self.args.models).split(",") if item.strip()]
        unknown = [item for item in values if item not in MODELS]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        return values

    def write_manifest(self) -> None:
        payload = {
            "run_id": self.args.run_id,
            "run_root": str(self.run_root),
            "dataset": self.args.dataset,
            "mode": "direct_baseline_only_dynamic_8gpu",
            "models": self.model_keys(),
            "gpus": self.gpu_ids(),
            "split_json": str(self.args.split_json),
            "data_root": str(self.args.data_root),
            "shard_size": self.args.shard_size,
            "port_base": self.args.port_base,
            "failures": self.failures,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (self.run_root / "run_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def test_count(self) -> int:
        split_payload, _ = load_split_payload(split_json=self.args.split_json, data_root=self.args.data_root)
        return len(split_payload["test_case_indices"])

    def tasks_for_model(self, model: str) -> list[ShardTask]:
        count = self.test_count()
        tasks = [
            ShardTask(model=model, shard_id=shard_id, offset=offset, limit=min(self.args.shard_size, count - offset))
            for shard_id, offset in enumerate(range(0, count, self.args.shard_size))
        ]
        if self.args.max_shards_per_model:
            tasks = tasks[: int(self.args.max_shards_per_model)]
        return tasks

    def done_path(self, task: ShardTask) -> Path:
        return self.shards_root / task.model / f"shard_{task.shard_id:05d}" / "DONE.json"

    def run(self) -> None:
        self.write_manifest()
        if self.args.dry_run:
            for model in self.model_keys():
                self.log(f"dry-run {model}: {len(self.tasks_for_model(model))} shards")
            return
        for model in self.model_keys():
            if self.stop_event.is_set():
                break
            self.run_model(model)
            self.merge_model(model)
            self.write_manifest()
        self.write_manifest()
        if self.failures:
            raise SystemExit(2)

    def run_model(self, model: str) -> None:
        tasks = [task for task in self.tasks_for_model(model) if not (self.args.resume and self.done_path(task).exists())]
        pending: queue.Queue[ShardTask] = queue.Queue()
        for task in tasks:
            pending.put(task)
        self.log(f"{model}: {pending.qsize()} pending shards on GPUs {self.gpu_ids()}")
        if pending.empty():
            self.log(f"{model}: no pending shards; skipping service startup")
            return
        threads = [
            threading.Thread(target=self.worker_loop, args=(model, gpu, self.args.port_base + gpu, pending), daemon=True)
            for gpu in self.gpu_ids()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log(f"{model}: stage complete")

    def worker_loop(self, model: str, gpu: int, port: int, pending: queue.Queue[ShardTask]) -> None:
        pid_path: Path | None = None
        try:
            pid_path = self.start_service(model=model, gpu=gpu, port=port)
            self.verify_model_id(model=model, port=port)
            while not self.stop_event.is_set():
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.run_task_with_retry(task=task, gpu=gpu, port=port)
                except Exception as exc:
                    failure = {"task": task.__dict__, "gpu": gpu, "port": port, "error": str(exc)}
                    with self.lock:
                        self.failures.append(failure)
                    self.log(f"FAILED {model} shard {task.shard_id:05d} on GPU{gpu}: {exc}")
                finally:
                    pending.task_done()
        finally:
            if pid_path is not None:
                self.stop_service(pid_path=pid_path, port=port)

    def run_task_with_retry(self, task: ShardTask, gpu: int, port: int) -> None:
        attempts = int(self.args.retries) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.verify_model_id(model=task.model, port=port)
                self.run_task(task=task, gpu=gpu, port=port, attempt=attempt)
                self.mark_done(task=task, gpu=gpu, port=port)
                return
            except Exception as exc:
                last_error = exc
                self.log(f"{task.model} shard {task.shard_id:05d} attempt {attempt}/{attempts} failed on GPU{gpu}: {exc}")
                time.sleep(5)
        raise RuntimeError(str(last_error))

    def start_service(self, *, model: str, gpu: int, port: int) -> Path:
        spec = MODELS[model]
        pid_path = self.pids_root / f"{model}_gpu{gpu}_{port}.pid"
        log_path = self.logs_root / f"service_{model}_gpu{gpu}_{port}.log"
        startup_log = self.logs_root / f"service_{model}_gpu{gpu}_{port}.startup.log"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PORT": str(port),
                "HOST": "127.0.0.1",
                "OPENAI_API_KEY": "EMPTY",
                "FORCE_RESTART": "1",
                "MAX_NUM_SEQS": "1",
                "LOG_FILE": str(log_path),
                "PID_FILE": str(pid_path),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "WAIT_SECONDS": str(self.args.service_timeout),
            }
        )
        if spec.get("max_model_len"):
            env["MAX_MODEL_LEN"] = spec["max_model_len"]
        if spec.get("gpu_memory_utilization"):
            env["GPU_MEMORY_UTILIZATION"] = spec["gpu_memory_utilization"]
        self.log(f"starting {model} on GPU{gpu} port {port}")
        with startup_log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / spec["start_script"]), str(PROJECT_ROOT)],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{model} startup rc={completed.returncode}; see {startup_log}")
        if not self.wait_ready(port=port, timeout=self.args.service_timeout):
            raise RuntimeError(f"{model} on port {port} did not become ready; see {startup_log}")
        return pid_path

    def wait_ready(self, *, port: int, timeout: int) -> bool:
        deadline = time.time() + int(timeout)
        while time.time() < deadline:
            try:
                req = Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer EMPTY"})
                with urlopen(req, timeout=5) as response:
                    if response.status < 500:
                        return True
            except Exception:
                time.sleep(5)
        return False

    def verify_model_id(self, *, model: str, port: int) -> None:
        expected = MODELS[model]["model_name"]
        req = Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer EMPTY"})
        with urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ids = [str(item.get("id", "")) for item in payload.get("data", [])]
        if expected not in ids:
            raise RuntimeError(f"Port {port} expected {expected}, got {ids}")

    def stop_service(self, *, pid_path: Path, port: int) -> None:
        pids: set[int] = set()
        if pid_path.exists():
            text = pid_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text.isdigit():
                pids.add(int(text))
        try:
            output = subprocess.check_output(
                ["bash", "-lc", f"ss -ltnp 'sport = :{int(port)}' 2>/dev/null | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p'"],
                text=True,
            )
            for item in output.split():
                if item.isdigit():
                    pids.add(int(item))
        except Exception:
            pass
        for pid in sorted(pids):
            subprocess.run(["bash", "-lc", f"pkill -TERM -P {pid} 2>/dev/null || true; kill -TERM {pid} 2>/dev/null || true"], check=False)
        time.sleep(6)
        for pid in sorted(pids):
            subprocess.run(["bash", "-lc", f"pkill -KILL -P {pid} 2>/dev/null || true; kill -KILL {pid} 2>/dev/null || true"], check=False)
        pid_path.unlink(missing_ok=True)

    def run_task(self, *, task: ShardTask, gpu: int, port: int, attempt: int) -> None:
        shard_root = self.shards_root / task.model / f"shard_{task.shard_id:05d}"
        output_root = shard_root / "eval"
        log_path = shard_root / f"gpu{gpu}_attempt{attempt}.log"
        shard_root.mkdir(parents=True, exist_ok=True)
        spec = MODELS[task.model]
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "OPENAI_API_KEY": "EMPTY",
                "OPENAI_MODEL": spec["model_name"],
                "DERMAGENT_POLICY_ROOT": str(self.args.policy_root),
                "DERMAGENT_SPLIT_STATE_ROOT": str(self.args.split_state_root),
                "DERMAGENT_DATA_ROOT": str(PROJECT_ROOT / "data"),
                "OPENAI_TIMEOUT": str(self.args.client_timeout),
            }
        )
        script = f"""
import json
from pathlib import Path
from agent.evaluation_protocol import EvaluationTargetSpec, run_evaluation_suite
from agent.policy_config import load_stable_policy
from integrations.openai_client import DermOpenAIClient

client = DermOpenAIClient(
    base_url='http://127.0.0.1:{port}/v1',
    api_key='EMPTY',
    model={spec['model_name']!r},
    timeout={float(self.args.client_timeout)!r},
    max_retries={int(self.args.client_max_retries)!r},
)
target = EvaluationTargetSpec(
    target_id='direct_baseline',
    label={task.model!r} + ' direct baseline',
    target_type='baseline',
    mode='baseline',
    description={f"Direct model baseline on {self.args.dataset} final 30 percent test split; no DermAgent workflow."!r},
)
suite = run_evaluation_suite(
    output_root=Path({str(output_root)!r}),
    data_root=Path({str(self.args.data_root)!r}),
    client=client,
    baseline_client=client,
    policy_config=load_stable_policy().to_dict(),
    target_specs=[target],
    limit={int(task.limit)!r},
    case_offset={int(task.offset)!r},
    seed=0,
    suite_label={task.model!r} + '_' + {self.args.dataset!r} + '_direct_baseline',
    data_split='test',
    split_json=Path({str(self.args.split_json)!r}),
    strict_frozen_eval=True,
)
result = suite['result_manifest']
summary = (result.get('target_results') or [{{}}])[0].get('summary', {{}})
payload = {{
    'model': {task.model!r},
    'model_name': {spec['model_name']!r},
    'gpu': {gpu!r},
    'port': {port!r},
    'shard_id': {task.shard_id!r},
    'offset': {task.offset!r},
    'limit': {task.limit!r},
    'summary': summary,
    'run_root': suite['run_root'],
    'evaluation_manifest_path': suite['evaluation_manifest_path'],
    'result_manifest_path': suite['result_manifest_path'],
}}
Path({str(shard_root / 'shard_result.json')!r}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\\n')
print(json.dumps(payload, ensure_ascii=False, indent=2))
"""
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(PYTHON), "-c", script],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"shard command rc={completed.returncode}; see {log_path}")

    def mark_done(self, *, task: ShardTask, gpu: int, port: int) -> None:
        result_path = self.shards_root / task.model / f"shard_{task.shard_id:05d}" / "shard_result.json"
        payload = {
            "task": task.__dict__,
            "gpu": gpu,
            "port": port,
            "result_path": str(result_path),
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.done_path(task).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self.log(f"DONE {task.model} shard {task.shard_id:05d} offset={task.offset} limit={task.limit} on GPU{gpu}")

    def merge_model(self, model: str) -> None:
        records: list[dict[str, Any]] = []
        shard_results: list[dict[str, Any]] = []
        for done_path in sorted((self.shards_root / model).glob("shard_*/DONE.json")):
            done = json.loads(done_path.read_text(encoding="utf-8"))
            result_path = Path(str(done["result_path"]))
            shard_result = json.loads(result_path.read_text(encoding="utf-8"))
            shard_results.append(shard_result)
            result_manifest = json.loads(Path(shard_result["result_manifest_path"]).read_text(encoding="utf-8"))
            artifacts = (result_manifest.get("target_results") or [{}])[0].get("artifacts", {})
            records_path = Path(str(artifacts.get("records_jsonl_path", "")))
            if records_path.exists():
                with records_path.open("r", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle if line.strip())
        summary = build_policy_summary(records)
        report = {
            "run_config": {
                "model": model,
                "model_name": MODELS[model]["model_name"],
                "dataset": self.args.dataset,
                "split_json": str(self.args.split_json),
                "data_root": str(self.args.data_root),
                "mode": "direct_baseline_only",
                "num_shards": len(shard_results),
            },
            "summary": summary,
            "shard_results": shard_results,
            "cases": records,
        }
        out = self.reports_root / model / f"{self.args.dataset}_direct_baseline_merged.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        self.log(
            f"merged {model}: cases={len(records)} top1={summary.get('top1', {}).get('rate')} "
            f"topk={summary.get('topk', {}).get('rate')} malignant={summary.get('malignant_recall', {}).get('rate')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct baselines dynamically across 8 GPUs.")
    parser.add_argument("--run-id", default=f"pad20_direct_8gpu_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset", default="pad20")
    parser.add_argument("--models", default="qwen,llama,skinvl,hulumed,medgemma,dermatollama")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port-base", type=int, default=8900)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--client-timeout", type=float, default=240.0)
    parser.add_argument("--client-max-retries", type=int, default=4)
    parser.add_argument("--service-timeout", type=int, default=720)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-shards-per-model", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/pad_ufes_20")
    parser.add_argument("--split-json", type=Path, default=PROJECT_ROOT / "paper_data/final_dataset_splits_20260509/splits/pad20_final_30_70_split.json")
    default_asset = (
        PROJECT_ROOT
        / "paper_data/final_30_70_large_runs/final_30_70_3x3_0p5to3_strat_no_doctor_20260510T113540Z"
        / "machine_0/assets/final_30_70_3x3_0p5to3_strat_no_doctor_20260510T113540Z_hulumed_pad20"
    )
    parser.add_argument("--policy-root", type=Path, default=default_asset / "policy")
    parser.add_argument("--split-state-root", type=Path, default=default_asset / "split_states")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = PROJECT_ROOT / "paper_data" / f"{args.dataset}_direct_runs" / args.run_id
    return args


def main() -> int:
    Pad20DirectRunner(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
