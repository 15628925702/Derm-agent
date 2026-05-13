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

DIRECT_MODELS = ("qwen", "llama", "skinvl", "hulumed", "medgemma", "dermatollama")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Pad20Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.direct_root = PROJECT_ROOT / "paper_data" / "pad20_direct_runs" / args.direct_run_id
        self.agent_root = (
            PROJECT_ROOT
            / "paper_data"
            / "final_30_70_large_runs"
            / args.agent_run_id
            / f"machine_{args.machine_id}"
        )
        self.pipeline_root = PROJECT_ROOT / "paper_data" / "pad20_pipeline_runs" / args.pipeline_id
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
                "direct_run_id": self.args.direct_run_id,
                "direct_root": str(self.direct_root),
                "agent_run_id": self.args.agent_run_id,
                "agent_root": str(self.agent_root),
                "updated_at": utc_now(),
            }
        )
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def direct_report_path(self, model: str) -> Path:
        return self.direct_root / "reports" / model / "pad20_direct_baseline_merged.json"

    def agent_report_path(self) -> Path:
        return self.agent_root / "reports" / "hulumed" / "pad20" / "compare_agent_vs_qwen_final_compare_test_merged.json"

    def report_case_count(self, path: Path) -> int:
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
        return all(self.report_case_count(self.direct_report_path(model)) >= self.args.test_cases for model in DIRECT_MODELS)

    def agent_complete(self) -> bool:
        return self.report_case_count(self.agent_report_path()) >= self.args.test_cases

    def active_processes(self, script_name: str, run_id: str) -> list[str]:
        completed = subprocess.run(
            ["pgrep", "-af", script_name],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        lines = []
        for line in completed.stdout.splitlines():
            if run_id in line and str(Path(__file__).name) not in line:
                lines.append(line)
        return lines

    def wait_for_current_direct(self) -> None:
        while not self.direct_complete():
            active = self.active_processes("run_pad20_direct_8gpu_dynamic.py", self.args.direct_run_id)
            if not active:
                return
            self.log(f"direct stage already running; waiting ({len(active)} process)")
            self.write_state(stage="direct", status="waiting_existing_direct", active_processes=active)
            time.sleep(self.args.poll_seconds)

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

    def run_direct_until_complete(self) -> None:
        self.wait_for_current_direct()
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
                "--resume",
            ]
            rc = self.run_command(cmd, f"direct_resume_attempt{attempts}.log")
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
                    "topk": (summary.get("topk") or {}).get("rate"),
                    "malignant_recall": (summary.get("malignant_recall") or {}).get("rate"),
                    "error_rate": (summary.get("error_rate") or {}).get("rate"),
                    "report": str(path),
                }
            )
        rows.sort(key=lambda item: (item.get("top1") or 0, item.get("topk") or 0, item.get("malignant_recall") or 0), reverse=True)
        json_path = self.pipeline_root / "direct_6model_summary.json"
        csv_path = self.pipeline_root / "direct_6model_summary.csv"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.write_state(stage="direct", status="summarized", direct_summary_json=str(json_path), direct_summary_csv=str(csv_path))
        self.log(f"direct summary written: {json_path}")

    def run_agent_until_complete(self) -> None:
        attempts = 0
        while not self.agent_complete():
            active = self.active_processes("run_final_30_70_6x5_dynamic.py", self.args.agent_run_id)
            if active:
                self.log(f"hulumed agent stage already running; waiting ({len(active)} process)")
                self.write_state(stage="agent", status="waiting_existing_agent", active_processes=active)
                time.sleep(self.args.poll_seconds)
                continue
            attempts += 1
            if attempts > self.args.max_stage_attempts:
                raise RuntimeError("hulumed agent stage did not complete after retries")
            self.write_state(stage="agent", status="running", attempt=attempts)
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
                "pad20",
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
        self.log("pipeline started")
        self.run_direct_until_complete()
        self.summarize_direct()
        self.run_agent_until_complete()
        self.write_state(stage="done", status="complete")
        self.log("pipeline complete")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Background PAD20 pipeline: 6 direct models, then hulumed PAD20 Agent.")
    parser.add_argument("--pipeline-id", default=f"pad20_direct_then_hulumed_agent_{timestamp}")
    parser.add_argument("--direct-run-id", default="pad20_direct_8gpu_full_20260512T173402Z")
    parser.add_argument("--agent-run-id", default=f"pad20_hulumed_agent_30_70_{timestamp}")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--machine-id", type=int, default=0)
    parser.add_argument("--test-cases", type=int, default=689)
    parser.add_argument("--direct-port-base", type=int, default=8900)
    parser.add_argument("--direct-shard-size", type=int, default=32)
    parser.add_argument("--bootstrap-shard-size", type=int, default=64)
    parser.add_argument("--compare-shard-size", type=int, default=32)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--max-stage-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-sleep-seconds", type=int, default=180)
    parser.add_argument("--client-timeout", type=float, default=240.0)
    parser.add_argument("--client-max-retries", type=int, default=4)
    parser.add_argument("--service-timeout", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    Pad20Pipeline(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
