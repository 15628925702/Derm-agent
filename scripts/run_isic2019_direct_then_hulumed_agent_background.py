from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DATASET = "isic2019"
DIRECT_MODELS = ("qwen", "llama", "skinvl", "hulumed", "medgemma", "dermatollama")
SPLIT_JSON = PROJECT_ROOT / "paper_data/final_dataset_splits_20260509/splits/isic2019_final_30_70_split.json"
DATA_ROOT = PROJECT_ROOT / "data/isic2019"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_split_counts() -> tuple[int, int]:
    payload = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    train_count = len(payload.get("final_experience_train_case_indices") or payload.get("train_case_indices") or [])
    test_count = len(payload.get("final_compare_test_case_indices") or payload.get("test_case_indices") or [])
    return train_count, test_count


class Isic2019Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.train_cases, self.test_cases = load_split_counts()
        self.direct_root = PROJECT_ROOT / "paper_data" / "isic2019_direct_runs" / args.direct_run_id
        self.agent_root = (
            PROJECT_ROOT
            / "paper_data"
            / "final_30_70_large_runs"
            / args.agent_run_id
            / f"machine_{args.machine_id}"
        )
        self.pipeline_root = PROJECT_ROOT / "paper_data" / "isic2019_pipeline_runs" / args.pipeline_id
        self.pipeline_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.pipeline_root / "pipeline.log"
        self.state_path = self.pipeline_root / "pipeline_state.json"

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_state(self, **updates: Any) -> None:
        state: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state.update(updates)
        state.update(
            {
                "pipeline_id": self.args.pipeline_id,
                "dataset": DATASET,
                "split_json": str(SPLIT_JSON),
                "train_cases": self.train_cases,
                "test_cases": self.test_cases,
                "topk_eval": "fixed_top3",
                "direct_run_id": self.args.direct_run_id,
                "direct_root": str(self.direct_root),
                "agent_run_id": self.args.agent_run_id,
                "agent_root": str(self.agent_root),
                "updated_at": utc_now(),
            }
        )
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def direct_report_path(self, model: str) -> Path:
        return self.direct_root / "reports" / model / "isic2019_direct_baseline_merged.json"

    def agent_report_path(self) -> Path:
        return self.agent_root / "reports" / "hulumed" / "isic2019" / "compare_agent_vs_qwen_final_compare_test_merged.json"

    @staticmethod
    def report_case_count(path: Path) -> int:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        summary = payload.get("summary") or {}
        if isinstance(summary.get("num_cases"), int):
            return int(summary["num_cases"])
        cases = payload.get("cases") or []
        return len(cases) if isinstance(cases, list) else 0

    def direct_complete(self) -> bool:
        return all(self.report_case_count(self.direct_report_path(model)) >= self.test_cases for model in DIRECT_MODELS)

    def agent_complete(self) -> bool:
        return self.report_case_count(self.agent_report_path()) >= self.test_cases

    @staticmethod
    def active_processes(script_name: str, run_id: str) -> list[str]:
        completed = subprocess.run(
            ["pgrep", "-af", script_name],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return [line for line in completed.stdout.splitlines() if run_id in line]

    def run_command(self, cmd: list[str], log_name: str) -> int:
        log_path = self.pipeline_root / log_name
        self.log("RUN " + " ".join(cmd))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND: " + " ".join(cmd) + "\n")
            handle.flush()
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(f"\nRETURN_CODE: {completed.returncode}\n")
        self.log(f"DONE rc={completed.returncode} log={log_path}")
        return int(completed.returncode)

    def wait_for_existing(self, *, script_name: str, run_id: str, stage: str) -> None:
        while True:
            active = self.active_processes(script_name, run_id)
            if not active:
                return
            self.log(f"{stage} already running; waiting ({len(active)} process)")
            self.write_state(stage=stage, status="waiting_existing", active_processes=active)
            time.sleep(self.args.poll_seconds)

    def run_direct_until_complete(self) -> None:
        self.wait_for_existing(script_name="run_pad20_direct_8gpu_dynamic.py", run_id=self.args.direct_run_id, stage="direct")
        attempts = 0
        while not self.direct_complete():
            attempts += 1
            if attempts > self.args.max_stage_attempts:
                raise RuntimeError("direct stage did not complete after retries")
            self.write_state(stage="direct", status="running_resume", attempt=attempts)
            cmd = [
                str(PYTHON),
                "scripts/run_pad20_direct_8gpu_dynamic.py",
                "--run-id",
                self.args.direct_run_id,
                "--dataset",
                DATASET,
                "--data-root",
                str(DATA_ROOT),
                "--split-json",
                str(SPLIT_JSON),
                "--models",
                ",".join(DIRECT_MODELS),
                "--gpus",
                self.args.gpus,
                "--port-base",
                str(self.args.direct_port_base),
                "--shard-size",
                str(self.args.direct_shard_size),
                "--retries",
                str(self.args.task_retries),
                "--client-timeout",
                str(self.args.client_timeout),
                "--client-max-retries",
                str(self.args.client_max_retries),
                "--service-timeout",
                str(self.args.service_timeout),
                "--resume",
            ]
            rc = self.run_command(cmd, f"direct_attempt{attempts}.log")
            if rc == 0 and self.direct_complete():
                break
            self.log("direct stage incomplete after attempt; sleeping before resume")
            time.sleep(self.args.retry_sleep_seconds)
        self.write_state(stage="direct", status="complete")

    def summarize_direct(self) -> None:
        rows: list[dict[str, Any]] = []
        for model in DIRECT_MODELS:
            path = self.direct_report_path(model)
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary") or {}
            rows.append(
                {
                    "model": model,
                    "cases": summary.get("num_cases"),
                    "top1": (summary.get("top1") or {}).get("rate"),
                    "top3": (summary.get("topk") or {}).get("rate"),
                    "malignant_recall": (summary.get("malignant_recall") or {}).get("rate"),
                    "error_rate": (summary.get("error_rate") or {}).get("rate"),
                    "report": str(path),
                }
            )
        rows.sort(key=lambda item: (item.get("top1") or 0, item.get("top3") or 0, item.get("malignant_recall") or 0), reverse=True)
        json_path = self.pipeline_root / "direct_6model_summary_top3.json"
        csv_path = self.pipeline_root / "direct_6model_summary_top3.csv"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.write_state(stage="direct", status="summarized", direct_summary_json=str(json_path), direct_summary_csv=str(csv_path))
        self.log(f"direct summary written: {json_path}")

    def run_agent_until_complete(self) -> None:
        self.wait_for_existing(script_name="run_final_30_70_6x5_dynamic.py", run_id=self.args.agent_run_id, stage="agent")
        attempts = 0
        while not self.agent_complete():
            attempts += 1
            if attempts > self.args.max_stage_attempts:
                raise RuntimeError("hulumed agent stage did not complete after retries")
            self.write_state(stage="agent", status="running_resume", attempt=attempts)
            cmd = [
                str(PYTHON),
                "scripts/run_final_30_70_6x5_dynamic.py",
                "--run-id",
                self.args.agent_run_id,
                "--machine-count",
                "1",
                "--machine-id",
                str(self.args.machine_id),
                "--models",
                "hulumed",
                "--datasets",
                DATASET,
                "--bootstrap-gpus",
                self.args.gpus,
                "--compare-gpus",
                self.args.gpus,
                "--doctor-gpus",
                self.args.gpus,
                "--bootstrap-shard-size",
                str(self.args.bootstrap_shard_size),
                "--compare-shard-size",
                str(self.args.compare_shard_size),
                "--client-timeout",
                str(self.args.client_timeout),
                "--client-max-retries",
                str(self.args.client_max_retries),
                "--service-timeout",
                str(self.args.service_timeout),
                "--disable-physician-evidence-summary",
                "--resume",
            ]
            rc = self.run_command(cmd, f"hulumed_agent_attempt{attempts}.log")
            if rc == 0 and self.agent_complete():
                break
            self.log("hulumed agent stage incomplete after attempt; sleeping before resume")
            time.sleep(self.args.retry_sleep_seconds)
        self.write_state(stage="agent", status="complete", agent_report=str(self.agent_report_path()))

    def run(self) -> None:
        self.write_state(stage="starting", status="running")
        self.log(
            f"pipeline started: direct test={self.test_cases}, "
            f"hulumed agent train={self.train_cases} test={self.test_cases}, topk=fixed_top3"
        )
        self.run_direct_until_complete()
        self.summarize_direct()
        self.run_agent_until_complete()
        self.write_state(stage="done", status="complete")
        self.log("pipeline complete")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Background ISIC2019 pipeline: 6 direct models, then Hulu-Med Agent full 70/30.")
    parser.add_argument("--pipeline-id", default=f"isic2019_direct_then_hulumed_agent_{timestamp}")
    parser.add_argument("--direct-run-id", default=f"isic2019_direct_6models_30test_{timestamp}")
    parser.add_argument("--agent-run-id", default=f"isic2019_hulumed_agent_70train_30test_{timestamp}")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--machine-id", type=int, default=0)
    parser.add_argument("--direct-port-base", type=int, default=8900)
    parser.add_argument("--direct-shard-size", type=int, default=32)
    parser.add_argument("--bootstrap-shard-size", type=int, default=64)
    parser.add_argument("--compare-shard-size", type=int, default=32)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--max-stage-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-sleep-seconds", type=int, default=180)
    parser.add_argument("--client-timeout", type=float, default=300.0)
    parser.add_argument("--client-max-retries", type=int, default=4)
    parser.add_argument("--service-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    Isic2019Pipeline(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
