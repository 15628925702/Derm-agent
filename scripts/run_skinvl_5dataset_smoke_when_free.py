from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
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

from agent.label_space import canonicalize_label
from dataio.case_loader import load_case_by_index
from integrations.openai_client import DermOpenAIClient


DATASETS: dict[str, dict[str, str]] = {
    "ham10000": {
        "data_root": "data/ham10000",
        "split_json": "paper_data/final_dataset_splits_20260509/splits/ham10000_final_30_70_split.json",
    },
    "isic2019": {
        "data_root": "data/isic2019",
        "split_json": "paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json",
    },
    "pad20": {
        "data_root": "data/pad_ufes_20",
        "split_json": "paper_data/final_dataset_splits_20260509/splits/pad20_final_30_70_split.json",
    },
    "scin": {
        "data_root": "data/scin",
        "split_json": "paper_data/final_dataset_splits_20260509/splits/scin_final_30_70_split.json",
    },
    "sd198": {
        "data_root": "data/sd198/sd-198",
        "split_json": "paper_data/final_dataset_splits_20260509/splits/sd198_final_30_70_split.json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SkinVLSmokeRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = args.output_root.resolve()
        self.log_dir = self.run_root / "logs"
        self.pid_file = self.run_root / "skinvl_server.pid"
        self.log_path = self.run_root / "runner.log"
        self.results_path = self.run_root / "skinvl_5dataset_20case_results.jsonl"
        self.summary_path = self.run_root / "skinvl_5dataset_20case_summary.json"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_was_paused = False

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def pause_pipeline(self) -> None:
        pid = int(self.args.pause_pipeline_pid or 0)
        if pid <= 0:
            return
        try:
            os.kill(pid, signal.SIGSTOP)
            self.pipeline_was_paused = True
            self.log(f"paused pipeline supervisor pid={pid}")
        except ProcessLookupError:
            self.log(f"pipeline supervisor pid={pid} is not running; skip pause")

    def resume_pipeline(self) -> None:
        pid = int(self.args.pause_pipeline_pid or 0)
        if pid <= 0 or not self.pipeline_was_paused:
            return
        try:
            os.kill(pid, signal.SIGCONT)
            self.log(f"resumed pipeline supervisor pid={pid}")
        except ProcessLookupError:
            self.log(f"pipeline supervisor pid={pid} is not running; skip resume")

    @staticmethod
    def process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.exists():
            try:
                parts = proc_stat.read_text(encoding="utf-8").split()
                if len(parts) >= 3 and parts[2] == "Z":
                    return False
            except Exception:
                pass
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def wait_for_process_exit(self) -> None:
        pid = int(self.args.wait_process_pid or 0)
        if pid <= 0:
            return
        while self.process_alive(pid):
            self.log(f"waiting for current direct process pid={pid} to finish before SkinVL smoke")
            time.sleep(self.args.poll_seconds)
        self.log(f"direct process pid={pid} finished")

    @staticmethod
    def gpu_snapshot() -> list[dict[str, int]]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        rows: list[dict[str, int]] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                rows.append({"index": int(parts[0]), "memory_used": int(parts[1]), "utilization": int(parts[2])})
            except ValueError:
                continue
        return rows

    def wait_for_free_gpu(self) -> int:
        allowed = [int(item) for item in str(self.args.gpus).split(",") if item.strip()]
        deadline = time.time() + float(self.args.max_wait_seconds)
        while time.time() < deadline:
            candidates = [
                row
                for row in self.gpu_snapshot()
                if row["index"] in allowed
                and row["memory_used"] <= self.args.max_gpu_memory_mb
                and row["utilization"] <= self.args.max_gpu_utilization
            ]
            if candidates:
                chosen = sorted(candidates, key=lambda row: (row["memory_used"], row["utilization"], row["index"]))[0]
                self.log(f"selected GPU{chosen['index']} for SkinVL smoke: {chosen}")
                return int(chosen["index"])
            self.log("no free GPU yet for SkinVL smoke")
            time.sleep(self.args.poll_seconds)
        raise TimeoutError("timed out waiting for a free GPU")

    def start_skinvl_service(self, gpu: int) -> None:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PORT": str(self.args.port),
                "HOST": "127.0.0.1",
                "OPENAI_API_KEY": "EMPTY",
                "LOG_DIR": str(self.log_dir),
                "LOG_FILE": str(self.log_dir / "skinvl_server.log"),
                "PID_FILE": str(self.pid_file),
                "FORCE_RESTART": "1",
                "CONV_MODE": "mistral_instruct",
                "MAX_NEW_TOKENS_DEFAULT": str(self.args.max_new_tokens),
                "WAIT_SECONDS": str(self.args.service_timeout),
            }
        )
        cmd = ["bash", "scripts/start_skinvl_server.sh", str(PROJECT_ROOT)]
        self.log("starting SkinVL service: " + " ".join(cmd))
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (self.log_dir / "skinvl_startup.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"SkinVL service failed to start rc={completed.returncode}")
        self.wait_for_service()

    def wait_for_service(self) -> None:
        deadline = time.time() + float(self.args.service_timeout)
        url = f"http://127.0.0.1:{self.args.port}/v1/models"
        while time.time() < deadline:
            try:
                request = Request(url, headers={"Authorization": "Bearer EMPTY"})
                with urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        self.log(f"SkinVL service ready at {url}")
                        return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"SkinVL service did not become ready: {url}")

    def stop_skinvl_service(self) -> None:
        if not self.pid_file.exists():
            return
        try:
            pid = int(self.pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return
        if pid <= 0:
            return
        self.log(f"stopping SkinVL service pid={pid}")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            time.sleep(3)
            if not self.process_alive(pid):
                break

    @staticmethod
    def split_indices(dataset: str, limit: int) -> list[int]:
        split_path = PROJECT_ROOT / DATASETS[dataset]["split_json"]
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        indices = (
            payload.get("final_compare_test_case_indices")
            or payload.get("test_case_indices")
            or payload.get("test")
            or []
        )
        if not indices:
            raise ValueError(f"split has no test indices: {split_path}")
        return [int(item) for item in indices[:limit]]

    def load_smoke_cases(self) -> list[tuple[str, Any]]:
        cases: list[tuple[str, Any]] = []
        for dataset in self.args.datasets.split(","):
            dataset = dataset.strip()
            if not dataset:
                continue
            data_root = PROJECT_ROOT / DATASETS[dataset]["data_root"]
            for case_index in self.split_indices(dataset, self.args.cases_per_dataset):
                case_input = load_case_by_index(case_index=case_index, data_root=data_root)
                cases.append((dataset, case_input))
        return cases

    @staticmethod
    def is_echo_payload(payload: dict[str, Any]) -> bool:
        text = " ".join(
            str(payload.get(key, "") or "")
            for key in ("final_diagnosis", "rationale", "raw_text", "parse_warning")
        )
        echo_markers = (
            "Your previous answer was not valid JSON",
            "Allowed canonical label IDs",
            "The field `final_diagnosis`",
            "Return valid JSON only",
        )
        return any(marker in text for marker in echo_markers)

    def run_cases(self) -> list[dict[str, Any]]:
        client = DermOpenAIClient(
            base_url=f"http://127.0.0.1:{self.args.port}/v1",
            api_key="EMPTY",
            model="SkinVL-MM",
            timeout=float(self.args.client_timeout),
            max_retries=0,
        )
        rows: list[dict[str, Any]] = []
        self.results_path.write_text("", encoding="utf-8")
        for ordinal, (dataset, case_input) in enumerate(self.load_smoke_cases(), start=1):
            started = time.time()
            self.log(f"case {ordinal}: {dataset}/{case_input.case_id}")
            error = ""
            payload: dict[str, Any] = {}
            try:
                payload = client.baseline_diagnosis(case_input)
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
            predicted = str(payload.get("final_diagnosis", "")).strip()
            canonical_pred = canonicalize_label(
                predicted,
                label_space_id=getattr(case_input, "label_space_id", None),
                dataset_name=getattr(case_input, "dataset_name", None),
                metadata=getattr(case_input, "metadata", None),
            )
            canonical_gt = canonicalize_label(
                getattr(case_input, "reference_label", None) or getattr(case_input, "label", None),
                label_space_id=getattr(case_input, "label_space_id", None),
                dataset_name=getattr(case_input, "dataset_name", None),
                metadata=getattr(case_input, "metadata", None),
            )
            row = {
                "dataset": dataset,
                "case_id": case_input.case_id,
                "ground_truth": canonical_gt,
                "final_diagnosis": predicted,
                "canonical_prediction": canonical_pred or "",
                "differential_diagnoses": payload.get("differential_diagnoses", []),
                "parse_warning": payload.get("parse_warning", ""),
                "confidence": payload.get("confidence", ""),
                "echo_detected": self.is_echo_payload(payload),
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
                "rationale_preview": str(payload.get("rationale", ""))[:300],
            }
            rows.append(row)
            with self.results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rows

    def write_summary(self, rows: list[dict[str, Any]]) -> None:
        by_dataset: dict[str, dict[str, Any]] = {}
        for dataset in self.args.datasets.split(","):
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            pred_counts = Counter(str(row.get("canonical_prediction") or "UNPARSED") for row in dataset_rows)
            by_dataset[dataset] = {
                "cases": len(dataset_rows),
                "nonempty_final": sum(1 for row in dataset_rows if str(row.get("final_diagnosis") or "").strip()),
                "canonical_parsed": sum(1 for row in dataset_rows if str(row.get("canonical_prediction") or "").strip()),
                "echo_detected": sum(1 for row in dataset_rows if row.get("echo_detected")),
                "errors": sum(1 for row in dataset_rows if row.get("error")),
                "prediction_counts": dict(pred_counts.most_common()),
            }
        payload = {
            "run_id": self.args.run_id,
            "created_at": utc_now(),
            "model": "SkinVL-MM",
            "cases_per_dataset": self.args.cases_per_dataset,
            "datasets": by_dataset,
            "results_jsonl": str(self.results_path),
        }
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.log(f"summary written: {self.summary_path}")

    def run(self) -> None:
        self.pause_pipeline()
        try:
            self.wait_for_process_exit()
            gpu = self.wait_for_free_gpu()
            self.start_skinvl_service(gpu)
            try:
                rows = self.run_cases()
                self.write_summary(rows)
            finally:
                self.stop_skinvl_service()
        finally:
            self.resume_pipeline()


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Wait for a GPU, then run SkinVL 20-case smoke across final datasets.")
    parser.add_argument("--run-id", default=f"skinvl_5dataset_smoke_{timestamp}")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "paper_data/diagnostic_runs" / f"skinvl_5dataset_smoke_{timestamp}")
    parser.add_argument("--datasets", default="ham10000,isic2019,pad20,scin,sd198")
    parser.add_argument("--cases-per-dataset", type=int, default=20)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--pause-pipeline-pid", type=int, default=0)
    parser.add_argument("--wait-process-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=86400)
    parser.add_argument("--max-gpu-memory-mb", type=int, default=2000)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--client-timeout", type=float, default=300.0)
    parser.add_argument("--service-timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    SkinVLSmokeRunner(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
