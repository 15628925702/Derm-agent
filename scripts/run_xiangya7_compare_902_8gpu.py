#!/usr/bin/env python3
"""
Launch 8-GPU parallel compare for xiangya_7class full test set (902 cases).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/zhongnan/miniconda3/envs/dermagent-6x6/bin/python")

DATASET = "xiangya_7class"
EXPERIMENT_ID = "xiangya_7class_hulumed_tuned_v3"
DATA_ROOT = PROJECT_ROOT / "data" / "xiangya" / "xiangya_selected_7class_3000"
POLICY_ROOT = PROJECT_ROOT / "state" / "dataset_adaptation" / "xiangya_7class_hulumed_relaxed50_v1" / "policy"
SPLIT_STATE_ROOT = PROJECT_ROOT / "state" / "dataset_adaptation" / "xiangya_7class_hulumed_relaxed50_v1" / "split_states"
POLICY_CONFIG = POLICY_ROOT / "current_stable_policy.json"
SPLIT_JSON = PROJECT_ROOT / "outputs" / "dataset_adaptation" / "xiangya_7class_hulumed_relaxed50_v1" / "xiangya_7class_split.json"
AGENT_WORKFLOW = "hulumed__xiangya_7class__retrieval_open_v1"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "xiangya7_hulumed_tuned_v3" / "compare902_8gpu_correct_workflow"
COMPARE_SHARDS_ROOT = OUTPUT_ROOT / "compare_shards"

PORTS = [8013, 8014, 8015, 8016, 8017, 8018, 8019, 8020]
TOTAL_CASES = 902
NUM_SHARDS = 8


def launch_shard(shard_id: int, offset: int, limit: int, port: int) -> subprocess.Popen:
    """Launch a single compare shard."""
    shard_output = COMPARE_SHARDS_ROOT / f"shard_{shard_id:05d}"
    shard_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "compare_agent_vs_qwen.py"),
        "--data-root", str(DATA_ROOT),
        "--limit", str(limit),
        "--case-offset", str(offset),
        "--data-split", "test",
        "--split-json", str(SPLIT_JSON),
        "--output-dir", str(shard_output),
        "--policy-config", str(POLICY_CONFIG),
        "--policy-label", "xiangya_7class_hulumed_tuned_v3",
        "--agent-workflow", AGENT_WORKFLOW,
        "--agent-base-url", f"http://127.0.0.1:{port}/v1",
        "--baseline-base-url", f"http://127.0.0.1:{port}/v1",
    ]

    env = {
        "DERMAGENT_POLICY_ROOT": str(POLICY_ROOT),
        "DERMAGENT_SPLIT_STATE_ROOT": str(SPLIT_STATE_ROOT),
        "PYTHONPATH": str(PROJECT_ROOT),
    }

    log_file = shard_output / "runner.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    print(f"[shard {shard_id}] PID={proc.pid} offset={offset} limit={limit} port={port}")
    return proc


def main():
    """Launch all shards and wait for completion."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARE_SHARDS_ROOT.mkdir(parents=True, exist_ok=True)

    # Calculate shard offsets and limits
    base_limit = TOTAL_CASES // NUM_SHARDS
    remainder = TOTAL_CASES % NUM_SHARDS

    shards = []
    offset = 0
    for shard_id in range(NUM_SHARDS):
        limit = base_limit + (1 if shard_id < remainder else 0)
        shards.append({
            "shard_id": shard_id,
            "offset": offset,
            "limit": limit,
            "port": PORTS[shard_id],
        })
        offset += limit

    # Write manifest
    manifest_started = {
        "started": [
            {
                "pid": None,  # Will be filled after launch
                "out_dir": str(COMPARE_SHARDS_ROOT / f"shard_{s['shard_id']:05d}"),
                "limit": s["limit"],
                "offset": s["offset"],
                "agent_workflow": AGENT_WORKFLOW,
            }
            for s in shards
        ],
        "shards": [str(COMPARE_SHARDS_ROOT / f"shard_{s['shard_id']:05d}") for s in shards],
        "agent_workflow": AGENT_WORKFLOW,
    }

    manifest_path = OUTPUT_ROOT / "compare_manifest.started.json"
    manifest_path.write_text(json.dumps(manifest_started, indent=2))

    # Launch all shards
    procs = []
    for shard in shards:
        proc = launch_shard(shard["shard_id"], shard["offset"], shard["limit"], shard["port"])
        procs.append(proc)
        manifest_started["started"][shard["shard_id"]]["pid"] = proc.pid
        time.sleep(2)  # Stagger launches slightly

    # Update manifest with PIDs
    manifest_path.write_text(json.dumps(manifest_started, indent=2))

    print(f"\n[master] Launched {NUM_SHARDS} shards, waiting for completion...")

    # Wait for all to complete
    failed = []
    for i, proc in enumerate(procs):
        returncode = proc.wait()
        if returncode != 0:
            failed.append({
                "out_dir": str(COMPARE_SHARDS_ROOT / f"shard_{i:05d}"),
                "returncode": returncode,
                "limit": shards[i]["limit"],
                "offset": shards[i]["offset"],
            })
            print(f"[shard {i}] FAILED with returncode {returncode}")
        else:
            print(f"[shard {i}] completed successfully")

    # Write final manifest
    manifest_final = {
        **manifest_started,
        "failed": failed,
    }
    manifest_final_path = OUTPUT_ROOT / "compare_manifest.json"
    manifest_final_path.write_text(json.dumps(manifest_final, indent=2))

    if failed:
        print(f"\n[master] {len(failed)} shard(s) failed")
        sys.exit(1)
    else:
        print(f"\n[master] All {NUM_SHARDS} shards completed successfully")


if __name__ == "__main__":
    main()
