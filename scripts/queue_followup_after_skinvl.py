from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.model_workflow_router import (  # noqa: E402
    execution_overrides_for_run_agent,
    get_model_workflow_overrides,
)


CURRENT_MODELS = ("qwen", "medgemma", "skinvl")
CURRENT_DATASETS = ("pad20", "scin", "ham10000")
SKINVL_EXPECTED_SHARDS = {"pad20": 22, "scin": 29, "ham10000": 94}
SKINVL_EXPECTED_CASES = {"pad20": 689, "scin": 918, "ham10000": 3004}
FOLLOWUP_MODELS = ("hulumed",)
FOLLOWUP_MODEL_NAMES = {"hulumed": "Hulu-Med-7B"}
FOLLOWUP_DATASETS = ("pad20", "scin", "ham10000")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = args.output_root.resolve()
        self.log_path = args.log_path.resolve()
        self.state_path = args.state_path.resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_state(self, **payload: Any) -> None:
        state = {"updated_at": utc_now(), **payload}
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def wait_for_pid_exit(self, pid: int) -> None:
        self.log(f"waiting for upstream SkinVL scheduler pid={pid}")
        while self.pid_alive(pid):
            self.write_state(stage="waiting_for_skinvl", upstream_pid=pid)
            time.sleep(self.args.poll_seconds)
        self.log(f"upstream SkinVL scheduler exited pid={pid}")

    def validate_model_dataset_routes(self) -> None:
        self.log("preflight: validating follow-up workflow routing")
        failures: list[str] = []
        for model in FOLLOWUP_MODELS:
            model_name = FOLLOWUP_MODEL_NAMES[model]
            for dataset in FOLLOWUP_DATASETS:
                overrides = get_model_workflow_overrides(model_name, dataset_name=dataset)
                cell = str(overrides.get("workflow_cell_id", "")).strip()
                execution = execution_overrides_for_run_agent(overrides)
                if not cell:
                    failures.append(f"{model}/{dataset}: missing workflow_cell_id")
                if overrides.get("skip_experience_retrieval"):
                    failures.append(f"{model}/{dataset}: skip_experience_retrieval=true")
                if execution.get("enable_experience_retrieval") is False:
                    failures.append(f"{model}/{dataset}: enable_experience_retrieval=false")
                if execution.get("enable_skill_retrieval") is False:
                    failures.append(f"{model}/{dataset}: enable_skill_retrieval=false")
                self.log(
                    "route "
                    f"{model}/{dataset}: cell={cell or 'MISSING'} "
                    f"exec={json.dumps(execution, ensure_ascii=False)}"
                )
        if failures:
            raise RuntimeError("workflow routing preflight failed: " + "; ".join(failures))

    def validate_skinvl_completion(self) -> dict[str, Any]:
        self.log("validating completed SkinVL rerun")
        scheduler_dir = self.args.skinvl_scheduler_dir.resolve()
        if (scheduler_dir / "FAILED").exists():
            raise RuntimeError(f"SkinVL scheduler failure marker exists: {scheduler_dir / 'FAILED'}")

        summary: dict[str, Any] = {}
        for dataset, expected_shards in SKINVL_EXPECTED_SHARDS.items():
            done_paths = sorted((self.output_root / "skinvl" / dataset / "compare").glob("*.DONE.json"))
            if len(done_paths) != expected_shards:
                raise RuntimeError(
                    f"SkinVL {dataset} incomplete: {len(done_paths)}/{expected_shards} compare shards"
                )

            report_path = (
                self.output_root
                / "reports"
                / "skinvl"
                / dataset
                / "compare_agent_vs_qwen_final_compare_test_merged.json"
            )
            if not report_path.exists():
                raise RuntimeError(f"missing merged SkinVL report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cases = report.get("cases", [])
            if len(cases) != SKINVL_EXPECTED_CASES[dataset]:
                raise RuntimeError(
                    f"SkinVL {dataset} merged case count mismatch: "
                    f"{len(cases)}/{SKINVL_EXPECTED_CASES[dataset]}"
                )

            bad_routes: list[str] = []
            for done_path in done_paths:
                shard = int(done_path.name.split("_")[1].split(".")[0])
                log_path = self.output_root / "logs" / f"skinvl_{dataset}_compare_{shard:05d}.log"
                header = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5])
                if '"enable_experience_retrieval": true' not in header:
                    bad_routes.append(f"{dataset}/{shard:05d}: experience retrieval not true")
                if '"enable_skill_retrieval": true' not in header:
                    bad_routes.append(f"{dataset}/{shard:05d}: skill retrieval not true")
                if '"skip_experience_retrieval": true' in header or '"skip_specialist_skills": true' in header:
                    bad_routes.append(f"{dataset}/{shard:05d}: skip flag true")
            if bad_routes:
                raise RuntimeError("SkinVL routing validation failed: " + "; ".join(bad_routes[:10]))

            agent_vs_baseline = dict(report.get("summary", {}).get("agent_vs_baseline", {}) or {})
            summary[dataset] = {
                "shards": len(done_paths),
                "cases": len(cases),
                "top1_delta": agent_vs_baseline.get("top1_delta"),
                "topk_delta": agent_vs_baseline.get("topk_delta"),
                "malignant_recall_delta": agent_vs_baseline.get("malignant_recall_delta"),
            }
            self.log(f"SkinVL {dataset} validated: {summary[dataset]}")
        return summary

    def stop_skinvl_services(self) -> None:
        self.log("stopping SkinVL services before follow-up")
        subprocess.run(["bash", "-lc", "pkill -TERM -f 'serve_skinvl_openai.py' || true"], check=False)
        time.sleep(8)
        subprocess.run(["bash", "-lc", "pkill -KILL -f 'serve_skinvl_openai.py' || true"], check=False)
        time.sleep(3)

    def run_followup(self) -> None:
        self.validate_model_dataset_routes()
        self.stop_skinvl_services()
        self.write_state(stage="running_followup", followup_models=FOLLOWUP_MODELS, followup_datasets=FOLLOWUP_DATASETS)
        self.log("starting follow-up 4x3 expansion: add hulumed x pad20/scin/ham10000")
        cmd = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_final_30_70_6x5_dynamic.py"),
            "--run-id",
            self.args.run_id,
            "--output-root",
            str(self.output_root),
            "--machine-id",
            "0",
            "--machine-count",
            "1",
            "--models",
            ",".join(FOLLOWUP_MODELS),
            "--datasets",
            ",".join(FOLLOWUP_DATASETS),
            "--bootstrap-train-tenths",
            "0.5",
            "--disable-physician-evidence-summary",
            "--bootstrap-gpus",
            "0,1,2,3,4,5,6,7",
            "--compare-gpus",
            "0,1,2,3,4,5,6,7",
            "--bootstrap-shard-size",
            "64",
            "--compare-shard-size",
            "32",
            "--client-timeout",
            "240",
            "--client-max-retries",
            "4",
            "--service-timeout",
            "900",
            "--no-resume",
        ]
        with self.args.followup_runner_log.resolve().open("a", encoding="utf-8") as handle:
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
        if completed.returncode != 0:
            raise RuntimeError(f"follow-up runner failed rc={completed.returncode}; see {self.args.followup_runner_log}")
        self.validate_followup_completion()
        self.write_state(stage="complete", followup_models=FOLLOWUP_MODELS, followup_datasets=FOLLOWUP_DATASETS)
        self.log("follow-up 4x3 expansion completed and validated")

    def validate_followup_completion(self) -> None:
        self.log("validating follow-up hulumed reports and routing")
        expected_cases = {"pad20": 689, "scin": 918, "ham10000": 3004}
        expected_shards = {"pad20": 22, "scin": 29, "ham10000": 94}
        for dataset in FOLLOWUP_DATASETS:
            done_paths = sorted((self.output_root / "hulumed" / dataset / "compare").glob("*.DONE.json"))
            if len(done_paths) != expected_shards[dataset]:
                raise RuntimeError(f"hulumed/{dataset} incomplete: {len(done_paths)}/{expected_shards[dataset]}")
            report_path = (
                self.output_root
                / "reports"
                / "hulumed"
                / dataset
                / "compare_agent_vs_qwen_final_compare_test_merged.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if len(report.get("cases", [])) != expected_cases[dataset]:
                raise RuntimeError(f"hulumed/{dataset} merged case count mismatch")
            first_log = self.output_root / "logs" / f"hulumed_{dataset}_compare_00000.log"
            header = "\n".join(first_log.read_text(encoding="utf-8", errors="ignore").splitlines()[:5])
            if "model_workflow_routing" not in header:
                raise RuntimeError(f"hulumed/{dataset} missing routing log")
            if f"hulumed__{dataset}" not in header:
                raise RuntimeError(f"hulumed/{dataset} suspicious routing header: {header}")
            self.log(f"hulumed/{dataset} validated: cases={expected_cases[dataset]}")

    def run(self) -> None:
        self.write_state(stage="started")
        if self.args.wait_pid:
            self.wait_for_pid_exit(int(self.args.wait_pid))
        skinvl_summary = self.validate_skinvl_completion()
        self.write_state(stage="skinvl_validated", skinvl_summary=skinvl_summary)
        self.run_followup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue hulumed 4x3 after the fixed SkinVL rerun.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skinvl-scheduler-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--followup-runner-log", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    supervisor = Supervisor(parse_args())
    try:
        supervisor.run()
        return 0
    except Exception as exc:
        supervisor.log(f"FAILED: {exc}")
        supervisor.write_state(stage="failed", error=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
