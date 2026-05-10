from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import queue
import random
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

from scripts.build_final_dataset_splits import dataset_specs as final_dataset_specs
from scripts.build_final_dataset_splits import load_cases as load_final_split_cases


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


MODELS: dict[str, dict[str, str]] = {
    "qwen": {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "start_script": "scripts/start_qwen_server.sh",
        "max_model_len": "16384",
        "gpu_memory_utilization": "0.82",
    },
    "medgemma": {
        "model_name": "medgemma-4b-it",
        "start_script": "scripts/start_medgemma_server.sh",
        "max_model_len": "12288",
        "gpu_memory_utilization": "0.85",
    },
    "skinvl": {
        "model_name": "SkinVL-MM",
        "start_script": "scripts/start_skinvl_server.sh",
        "max_model_len": "",
        "gpu_memory_utilization": "",
    },
    "llama": {
        "model_name": "Llama-3.2-11B-Vision-Instruct",
        "start_script": "scripts/start_llama_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
    "hulumed": {
        "model_name": "Hulu-Med-7B",
        "start_script": "scripts/start_hulumed_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
    "dermatollama": {
        "model_name": "DermatoLlama-full",
        "start_script": "scripts/start_dermatollama_server.sh",
        "max_model_len": "8192",
        "gpu_memory_utilization": "0.90",
    },
}


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


@dataclass(frozen=True)
class Task:
    phase: str
    model: str
    dataset: str
    shard_id: int
    offset: int
    limit: int
    case_indices: tuple[int, ...] = ()


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = args.output_root.resolve()
        self.assets_root = self.output_root / "assets"
        self.logs_root = self.output_root / "logs"
        self.reports_root = self.output_root / "reports"
        self.paper_root = self.output_root / "paper_case_exports"
        self.doctor_root = PROJECT_ROOT / f"doctor_evidence_final_30_70_{args.run_id}"
        for path in (self.output_root, self.assets_root, self.logs_root, self.reports_root, self.paper_root, self.doctor_root):
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
            with (self.output_root / "runner.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def run(self) -> None:
        self.write_run_manifest()
        combos = self.assigned_combos()
        self.log(f"assigned combos: {', '.join(f'{m}/{d}' for m, d in combos)}")
        if not self.args.skip_bootstrap:
            bootstrap_tasks = self.build_bootstrap_tasks(combos)
            self.run_stage("bootstrap", bootstrap_tasks, self.parse_gpus(self.args.bootstrap_gpus))
            for model, dataset in combos:
                self.merge_bootstrap(model, dataset)
        if not self.args.skip_compare:
            for model, dataset in combos:
                self.promote_state(model, dataset)
            compare_tasks = self.build_compare_tasks(combos)
            compare_gpus = self.parse_gpus(self.args.compare_gpus)
            self.run_stage("compare", compare_tasks, compare_gpus)
            for model, dataset in combos:
                self.merge_compare(model, dataset)
            if self.args.enable_physician_evidence_summary:
                doctor_tasks = self.build_doctor_tasks(combos)
                self.run_doctor_stage(doctor_tasks, self.parse_gpus(self.args.doctor_gpus))
        self.write_run_manifest()
        if self.failures:
            raise SystemExit(2)

    def assigned_combos(self) -> list[tuple[str, str]]:
        models = [item for item in self.args.models.split(",") if item]
        datasets = [item for item in self.args.datasets.split(",") if item]
        combos = [(model, dataset) for model in models for dataset in datasets]
        selected = [
            combo for idx, combo in enumerate(combos)
            if idx % int(self.args.machine_count) == int(self.args.machine_id)
        ]
        for model, dataset in selected:
            if model not in MODELS:
                raise ValueError(f"Unknown model: {model}")
            if dataset not in DATASETS:
                raise ValueError(
                    f"Unknown dataset or missing final 30/70 split: {dataset}. "
                    "Current final_20260509 covers ham10000,isic2019,pad20,scin,sd198."
                )
        return selected

    def parse_gpus(self, text: str) -> list[int]:
        gpus = [int(item) for item in text.split(",") if item.strip()]
        if not gpus:
            raise ValueError("At least one GPU is required.")
        return gpus

    def combo_root(self, model: str, dataset: str) -> Path:
        return self.output_root / model / dataset

    def combo_assets(self, model: str, dataset: str) -> Path:
        return self.assets_root / f"{self.args.run_id}_{model}_{dataset}"

    def combo_policy_root(self, model: str, dataset: str) -> Path:
        return self.combo_assets(model, dataset) / "policy"

    def combo_split_state_root(self, model: str, dataset: str) -> Path:
        return self.combo_assets(model, dataset) / "split_states"

    def task_done_path(self, task: Task) -> Path:
        return self.combo_root(task.model, task.dataset) / task.phase / f"shard_{task.shard_id:05d}.DONE.json"

    def write_run_manifest(self) -> None:
        manifest = {
            "run_id": self.args.run_id,
            "created_or_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "machine_id": self.args.machine_id,
            "machine_count": self.args.machine_count,
            "output_root": str(self.output_root),
            "doctor_evidence_root": str(self.doctor_root),
            "datasets": {key: DATASETS[key] for key in self.args.datasets.split(",") if key},
            "models": {key: MODELS[key]["model_name"] for key in self.args.models.split(",") if key},
            "bootstrap_shard_size": self.args.bootstrap_shard_size,
            "compare_shard_size": self.args.compare_shard_size,
            "bootstrap_train_tenths": self.args.bootstrap_train_tenths,
            "physician_evidence_summary_enabled": bool(self.args.enable_physician_evidence_summary),
            "physician_evidence_summary_mode": "posthoc_after_compare",
            "failures": self.failures,
        }
        (self.output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def split_payload(self, dataset: str) -> dict[str, Any]:
        return json.loads((PROJECT_ROOT / DATASETS[dataset]["split_json"]).read_text(encoding="utf-8"))

    @staticmethod
    def stable_seed(*parts: Any) -> int:
        digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def label_by_case_index(self, dataset: str) -> dict[int, str]:
        spec_by_key = {spec.key: spec for spec in final_dataset_specs(PROJECT_ROOT / "data")}
        spec = spec_by_key.get(dataset)
        if spec is None:
            raise ValueError(f"No label-aware sampling spec available for dataset: {dataset}")
        labels: dict[int, str] = {}
        for case in load_final_split_cases(spec):
            labels[int(case.case_index)] = str(case.label).strip() or "UNKNOWN"
        return labels

    def sample_bootstrap_indices(
        self,
        *,
        dataset: str,
        train_indices: list[int],
        target_count: int,
        split_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        label_by_index = self.label_by_case_index(dataset)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in train_indices:
            grouped[label_by_index.get(int(index), "UNKNOWN")].append(int(index))
        grouped = {label: values for label, values in grouped.items() if values}
        if not grouped:
            raise ValueError(f"No labeled train cases available for bootstrap sampling: {dataset}")

        rng = random.Random(self.stable_seed(self.args.run_id, dataset, split_seed, "bootstrap_stratified_0p5"))
        for values in grouped.values():
            values.sort()
            rng.shuffle(values)

        label_order = sorted(grouped)
        target_count = max(1, min(int(target_count), sum(len(values) for values in grouped.values())))
        takes = {label: 0 for label in label_order}

        # Preserve rare-label coverage whenever the sample is large enough.
        remaining = target_count
        if target_count >= len(label_order):
            for label in label_order:
                takes[label] = 1
            remaining -= len(label_order)

        total_train = sum(len(grouped[label]) for label in label_order)
        quotas = {
            label: (len(grouped[label]) / total_train) * remaining
            for label in label_order
        }
        for label in label_order:
            extra = min(len(grouped[label]) - takes[label], int(quotas[label]))
            takes[label] += max(0, extra)

        while sum(takes.values()) < target_count:
            candidates = [
                label for label in label_order
                if takes[label] < len(grouped[label])
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda label: (
                    quotas[label] - int(quotas[label]),
                    len(grouped[label]) - takes[label],
                    label,
                ),
                reverse=True,
            )
            takes[candidates[0]] += 1

        selected_by_label = {
            label: grouped[label][: takes[label]]
            for label in label_order
            if takes[label] > 0
        }
        round_robin_labels = [label for label in label_order if selected_by_label.get(label)]
        rng.shuffle(round_robin_labels)
        selected: list[int] = []
        cursor = 0
        while len(selected) < target_count and round_robin_labels:
            label = round_robin_labels[cursor % len(round_robin_labels)]
            values = selected_by_label[label]
            if values:
                selected.append(values.pop(0))
            if not values:
                round_robin_labels = [item for item in round_robin_labels if selected_by_label.get(item)]
                cursor = 0
            else:
                cursor += 1

        train_counts = {label: len(grouped[label]) for label in label_order}
        sampled_counts = dict(sorted(Counter(label_by_index.get(index, "UNKNOWN") for index in selected).items()))
        manifest = {
            "strategy": "stratified_label_proportional_round_robin",
            "dataset": dataset,
            "split_seed": split_seed,
            "target_count": target_count,
            "selected_count": len(selected),
            "train_label_counts": dict(sorted(train_counts.items())),
            "sampled_label_counts": sampled_counts,
            "sampled_case_indices": selected,
        }
        return selected, manifest

    def ensure_combo_assets(self, model: str, dataset: str) -> None:
        policy_root = self.combo_policy_root(model, dataset)
        split_state_root = self.combo_split_state_root(model, dataset)
        if (policy_root / "current_stable_policy.json").exists() and (split_state_root / "train").exists():
            return
        subprocess.run(
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/manage_dataset_experiment_assets.py"),
                "init",
                "--experiment-id",
                self.combo_assets(model, dataset).name,
                "--assets-root",
                str(self.assets_root),
                "--outputs-root",
                str(self.combo_root(model, dataset) / "asset_outputs"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def build_bootstrap_tasks(self, combos: list[tuple[str, str]]) -> list[Task]:
        tasks: list[Task] = []
        for model, dataset in combos:
            self.ensure_combo_assets(model, dataset)
            payload = self.split_payload(dataset)
            indices = [int(item) for item in payload["train_case_indices"]]
            if self.args.bootstrap_train_tenths is not None:
                total_cases = len(payload["train_case_indices"]) + len(payload["test_case_indices"])
                keep = round(total_cases * float(self.args.bootstrap_train_tenths) / 10.0)
                keep = max(1, min(len(indices), int(keep)))
                indices, sample_manifest = self.sample_bootstrap_indices(
                    dataset=dataset,
                    train_indices=indices,
                    target_count=keep,
                    split_seed=int(payload.get("seed", 0) or 0),
                )
                sample_path = self.combo_root(model, dataset) / "bootstrap_sample_manifest.json"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                sample_path.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for shard_id, start in enumerate(range(0, len(indices), self.args.bootstrap_shard_size)):
                chunk = tuple(indices[start : start + self.args.bootstrap_shard_size])
                tasks.append(Task("bootstrap", model, dataset, shard_id, start, len(chunk), chunk))
        return tasks

    def build_compare_tasks(self, combos: list[tuple[str, str]]) -> list[Task]:
        tasks: list[Task] = []
        for model, dataset in combos:
            count = len(self.split_payload(dataset)["test_case_indices"])
            for shard_id, offset in enumerate(range(0, count, self.args.compare_shard_size)):
                limit = min(self.args.compare_shard_size, count - offset)
                tasks.append(Task("compare", model, dataset, shard_id, offset, limit))
        return tasks

    def build_doctor_tasks(self, combos: list[tuple[str, str]]) -> list[Task]:
        tasks: list[Task] = []
        for model, dataset in combos:
            report = self.merged_report_path(model, dataset)
            if not report.exists():
                continue
            cases = json.loads(report.read_text(encoding="utf-8")).get("cases", []) or []
            for shard_id, offset in enumerate(range(0, len(cases), self.args.doctor_shard_size)):
                limit = min(self.args.doctor_shard_size, len(cases) - offset)
                tasks.append(Task("doctor", model, dataset, shard_id, offset, limit))
        return tasks

    def run_stage(self, phase: str, tasks: list[Task], gpus: list[int]) -> None:
        pending = queue.Queue()
        for task in tasks:
            if self.args.resume and self.task_done_path(task).exists():
                continue
            pending.put(task)
        self.log(f"{phase}: {pending.qsize()} pending shards on GPUs {gpus}")
        threads = [
            threading.Thread(target=self.worker_loop, args=(phase, gpu, 8300 + gpu, pending), daemon=True)
            for gpu in gpus
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log(f"{phase}: stage complete")

    def run_doctor_stage(self, tasks: list[Task], gpus: list[int]) -> None:
        pending = queue.Queue()
        for task in tasks:
            if self.args.resume and self.task_done_path(task).exists():
                continue
            pending.put(task)
        self.log(f"doctor: {pending.qsize()} pending shards on GPUs {gpus}")
        threads = [
            threading.Thread(target=self.doctor_worker_loop, args=(gpu, 8400 + gpu, pending), daemon=True)
            for gpu in gpus
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log("doctor: stage complete")

    def doctor_worker_loop(self, gpu: int, port: int, pending: queue.Queue[Task]) -> None:
        service_pid_path: Path | None = None
        try:
            service_pid_path = self.start_service("qwen", gpu, port)
            while not self.stop_event.is_set():
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.run_doctor_task(task, port)
                    self.mark_done(task, gpu)
                except Exception as exc:
                    failure = {"task": task.__dict__, "gpu": gpu, "error": str(exc)}
                    with self.lock:
                        self.failures.append(failure)
                    self.log(f"FAILED doctor {task.model}/{task.dataset} shard {task.shard_id}: {exc}")
                finally:
                    pending.task_done()
        finally:
            if service_pid_path is not None:
                self.stop_service(service_pid_path, port)

    def worker_loop(self, phase: str, gpu: int, port: int, pending: queue.Queue[Task]) -> None:
        current_model = ""
        service_pid_path: Path | None = None
        while not self.stop_event.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            try:
                if current_model != task.model:
                    if service_pid_path is not None:
                        self.stop_service(service_pid_path, port)
                    service_pid_path = self.start_service(task.model, gpu, port)
                    current_model = task.model
                if phase == "bootstrap":
                    self.run_bootstrap_task(task, port)
                else:
                    self.run_compare_task(task, port)
                self.mark_done(task, gpu)
            except Exception as exc:
                failure = {"task": task.__dict__, "gpu": gpu, "error": str(exc)}
                with self.lock:
                    self.failures.append(failure)
                self.log(f"FAILED {phase} {task.model}/{task.dataset} shard {task.shard_id}: {exc}")
            finally:
                pending.task_done()
        if service_pid_path is not None:
            self.stop_service(service_pid_path, port)

    def start_service(self, model: str, gpu: int, port: int) -> Path:
        spec = MODELS[model]
        log_path = self.logs_root / f"service_gpu{gpu}_{model}_{port}.log"
        pid_path = self.output_root / "pids" / f"gpu{gpu}_{model}_{port}.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
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
            }
        )
        if spec.get("max_model_len"):
            env["MAX_MODEL_LEN"] = spec["max_model_len"]
        if spec.get("gpu_memory_utilization"):
            env["GPU_MEMORY_UTILIZATION"] = spec["gpu_memory_utilization"]
        self.log(f"starting {model} on GPU{gpu} port {port}")
        startup_log = self.logs_root / f"service_gpu{gpu}_{model}_{port}.startup.log"
        with startup_log.open("a", encoding="utf-8") as handle:
            subprocess.Popen(
                ["bash", str(PROJECT_ROOT / spec["start_script"]), str(PROJECT_ROOT)],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if not self.wait_ready(port, timeout=self.args.service_timeout):
            raise RuntimeError(f"{model} on GPU{gpu} port {port} did not become ready; see {log_path}")
        return pid_path

    def wait_ready(self, port: int, timeout: int) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                req = Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer EMPTY"})
                with urlopen(req, timeout=5) as response:
                    if response.status < 500:
                        return True
            except Exception:
                time.sleep(5)
        return False

    def stop_service(self, pid_path: Path, port: int) -> None:
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
        time.sleep(8)
        for pid in sorted(pids):
            subprocess.run(["bash", "-lc", f"pkill -KILL -P {pid} 2>/dev/null || true; kill -KILL {pid} 2>/dev/null || true"], check=False)
        pid_path.unlink(missing_ok=True)
        time.sleep(3)

    def run_bootstrap_task(self, task: Task, port: int) -> None:
        shard_root = self.combo_root(task.model, task.dataset) / "bootstrap_shards" / f"shard_{task.shard_id:05d}"
        split_state_root = shard_root / "split_states"
        output_dir = shard_root / "records"
        policy_root = self.combo_policy_root(task.model, task.dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
        (shard_root / "case_indices.json").write_text(json.dumps(list(task.case_indices), indent=2) + "\n", encoding="utf-8")
        env = self.base_env(task.model, task.dataset, port, policy_root, split_state_root)
        env.pop("DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY", None)
        for case_index in task.case_indices:
            cmd = [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/debug_single_case.py"),
                "--case-index",
                str(case_index),
                "--data-root",
                str(PROJECT_ROOT / DATASETS[task.dataset]["data_root"]),
                "--output-dir",
                str(output_dir),
                "--enable-writeback",
                "--data-split",
                "train",
                "--run-mode",
                f"{self.args.run_id}_{task.model}_{task.dataset}_bootstrap",
                "--client-base-url",
                f"http://127.0.0.1:{port}/v1",
                "--client-api-key",
                "EMPTY",
                "--client-model",
                MODELS[task.model]["model_name"],
                "--client-timeout",
                str(self.args.client_timeout),
                "--client-max-retries",
                str(self.args.client_max_retries),
            ]
            self.run_logged(cmd, env, self.logs_root / f"{task.model}_{task.dataset}_bootstrap_{task.shard_id:05d}.log")

    def run_compare_task(self, task: Task, port: int) -> None:
        output_dir = self.combo_root(task.model, task.dataset) / "compare_shards" / f"shard_{task.shard_id:05d}"
        policy_root = self.combo_policy_root(task.model, task.dataset)
        split_state_root = self.combo_split_state_root(task.model, task.dataset)
        env = self.base_env(task.model, task.dataset, port, policy_root, split_state_root)
        cmd = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/compare_agent_vs_qwen.py"),
            "--data-root",
            str(PROJECT_ROOT / DATASETS[task.dataset]["data_root"]),
            "--split-json",
            str(PROJECT_ROOT / DATASETS[task.dataset]["split_json"]),
            "--data-split",
            "test",
            "--limit",
            str(task.limit),
            "--case-offset",
            str(task.offset),
            "--output-dir",
            str(output_dir),
            "--agent-base-url",
            f"http://127.0.0.1:{port}/v1",
            "--baseline-base-url",
            f"http://127.0.0.1:{port}/v1",
            "--agent-api-key",
            "EMPTY",
            "--baseline-api-key",
            "EMPTY",
            "--agent-model",
            MODELS[task.model]["model_name"],
            "--baseline-model",
            MODELS[task.model]["model_name"],
            "--client-timeout",
            str(self.args.client_timeout),
            "--client-max-retries",
            str(self.args.client_max_retries),
            "--export-paper-case-data",
            "--paper-case-data-dir",
            str(self.paper_root / task.model / task.dataset / f"shard_{task.shard_id:05d}"),
        ]
        env.pop("DERMAGENT_ENABLE_PHYSICIAN_EVIDENCE_SUMMARY", None)
        self.run_logged(cmd, env, self.logs_root / f"{task.model}_{task.dataset}_compare_{task.shard_id:05d}.log")

    def run_doctor_task(self, task: Task, port: int) -> None:
        output_dir = self.doctor_root / task.model / task.dataset / f"shard_{task.shard_id:05d}"
        cmd = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/generate_doctor_evidence_posthoc.py"),
            "--compare-report",
            str(self.merged_report_path(task.model, task.dataset)),
            "--output-dir",
            str(output_dir),
            "--case-offset",
            str(task.offset),
            "--limit",
            str(task.limit),
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "EMPTY",
            "--model",
            "Qwen2.5-VL-7B-Instruct",
            "--detail",
            self.args.physician_evidence_detail,
            "--timeout",
            str(self.args.physician_summary_timeout),
            "--max-retries",
            str(self.args.physician_summary_max_retries),
        ]
        self.run_logged(cmd, os.environ.copy(), self.logs_root / f"{task.model}_{task.dataset}_doctor_{task.shard_id:05d}.log")

    def base_env(self, model: str, dataset: str, port: int, policy_root: Path, split_state_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "OPENAI_API_KEY": "EMPTY",
                "OPENAI_MODEL": MODELS[model]["model_name"],
                "DERMAGENT_POLICY_ROOT": str(policy_root),
                "DERMAGENT_SPLIT_STATE_ROOT": str(split_state_root),
                "DERMAGENT_DATA_ROOT": str(PROJECT_ROOT / "data"),
                "OPENAI_TIMEOUT": str(self.args.client_timeout),
            }
        )
        if dataset == "scin":
            env["DERMAGENT_SCIN_LABEL_SPACE_ID"] = "scin_grouped"
        if dataset == "sd198":
            env["DERMAGENT_SD198_LABEL_SPACE_ID"] = "sd198_grouped"
        return env

    def run_logged(self, cmd: list[str], env: dict[str, str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND: " + " ".join(cmd) + "\n")
            handle.flush()
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
            handle.write(f"\nRETURN_CODE: {completed.returncode}\n")
        if completed.returncode != 0:
            raise RuntimeError(f"command failed rc={completed.returncode}; see {log_path}")

    def mark_done(self, task: Task, gpu: int) -> None:
        payload = {
            "task": task.__dict__,
            "gpu": gpu,
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path = self.task_done_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def merge_bootstrap(self, model: str, dataset: str) -> None:
        self.log(f"merging bootstrap state for {model}/{dataset}")
        subprocess.run(
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/merge_experience_shards.py"),
                "--shard-root",
                str(self.combo_root(model, dataset) / "bootstrap_shards"),
                "--output-split-state-root",
                str(self.combo_split_state_root(model, dataset)),
                "--manifest",
                str(self.combo_root(model, dataset) / "bootstrap_merge_manifest.json"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def promote_state(self, model: str, dataset: str) -> None:
        self.log(f"promoting train state to test for {model}/{dataset}")
        subprocess.run(
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/manage_dataset_experiment_assets.py"),
                "promote-state",
                "--split-state-root",
                str(self.combo_split_state_root(model, dataset)),
                "--source-split",
                "train",
                "--target-splits",
                "test",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def merge_compare(self, model: str, dataset: str) -> None:
        glob_pattern = str(self.combo_root(model, dataset) / "compare_shards" / "shard_*" / "compare_agent_vs_qwen_*.json")
        out_report = self.merged_report_path(model, dataset)
        subprocess.run(
            [
                str(PYTHON),
                str(PROJECT_ROOT / "scripts/merge_final_compare_reports.py"),
                "--compare-report-glob",
                glob_pattern,
                "--output-report",
                str(out_report),
                "--paper-case-data-dir",
                str(self.paper_root / model / dataset / "merged"),
                "--export-stem",
                f"{model}_{dataset}_final_compare_test_cases",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def merged_report_path(self, model: str, dataset: str) -> Path:
        return self.reports_root / model / dataset / "compare_agent_vs_qwen_final_compare_test_merged.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic 8-GPU runner for final_20260509 30/70 full bootstrap+compare."
    )
    parser.add_argument("--run-id", type=str, default=f"final_30_70_6x5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--machine-id", type=int, default=0)
    parser.add_argument("--machine-count", type=int, default=2)
    parser.add_argument("--models", type=str, default="qwen,llama,skinvl,hulumed,medgemma,dermatollama")
    parser.add_argument("--datasets", type=str, default="ham10000,isic2019,pad20,scin,sd198")
    parser.add_argument("--bootstrap-gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--compare-gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--doctor-gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--bootstrap-train-tenths",
        type=float,
        default=None,
        help="If set, bootstrap only the first N train cases equivalent to this many tenths of the full split population; compare still uses the full test split.",
    )
    parser.add_argument("--bootstrap-shard-size", type=int, default=64)
    parser.add_argument("--compare-shard-size", type=int, default=32)
    parser.add_argument("--client-timeout", type=float, default=240.0)
    parser.add_argument("--client-max-retries", type=int, default=4)
    parser.add_argument("--service-timeout", type=int, default=600)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--enable-physician-evidence-summary", action="store_true", default=True)
    parser.add_argument("--disable-physician-evidence-summary", dest="enable_physician_evidence_summary", action="store_false")
    parser.add_argument("--doctor-shard-size", type=int, default=32)
    parser.add_argument("--physician-evidence-detail", type=str, default="detailed", choices=("brief", "detailed"))
    parser.add_argument("--physician-summary-timeout", type=float, default=240.0)
    parser.add_argument("--physician-summary-max-retries", type=int, default=3)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = PROJECT_ROOT / "paper_data" / "final_30_70_large_runs" / args.run_id / f"machine_{args.machine_id}"
    return args


def main() -> int:
    Runner(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
