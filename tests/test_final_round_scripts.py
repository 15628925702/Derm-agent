from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "bootstrap_pad20_train_cases.sh",
    PROJECT_ROOT / "scripts" / "run_dataset_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_pad20_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_isic2019_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_ham10000_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_scin_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_sd198_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_xiangya_sft_final_round.sh",
    PROJECT_ROOT / "scripts" / "run_all_final_rounds.sh",
    PROJECT_ROOT / "scripts" / "run_qwen_final_round_asset_rerun.sh",
]


def test_final_round_scripts_have_valid_bash_syntax() -> None:
    cmd = ["bash", "-n", *[str(path) for path in FINAL_SCRIPTS]]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def test_final_round_wrappers_support_dry_run() -> None:
    wrappers = [
        PROJECT_ROOT / "scripts" / "run_pad20_final_round.sh",
        PROJECT_ROOT / "scripts" / "run_isic2019_final_round.sh",
        PROJECT_ROOT / "scripts" / "run_ham10000_final_round.sh",
        PROJECT_ROOT / "scripts" / "run_scin_final_round.sh",
        PROJECT_ROOT / "scripts" / "run_sd198_final_round.sh",
        PROJECT_ROOT / "scripts" / "run_xiangya_sft_final_round.sh",
    ]
    for path in wrappers:
        env = os.environ.copy()
        env["DRY_RUN"] = "1"
        completed = subprocess.run(
            ["bash", str(path)],
            check=True,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert "[final-round] dataset" in completed.stdout
        assert "[dry-run]" in completed.stdout
        assert "[final-round] server timeout" in completed.stdout
        assert "[final-round] model wait" in completed.stdout
        assert "[final-round] chat check" in completed.stdout


def test_all_final_rounds_support_dry_run_and_print_totals() -> None:
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["CLEAN_FINAL_ROUND"] = "1"
    completed = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_all_final_rounds.sh")],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "[all-final] bootstrap total" in completed.stdout
    assert "[all-final] compare total" in completed.stdout
    assert "[all-final] total case steps" in completed.stdout
    assert "[all-final] >>> 1/6 PAD-UFES-20" in completed.stdout
    assert "[all-final] >>> 6/6 HAM10000" in completed.stdout


def test_qwen_asset_rerun_preserves_grouped_label_spaces() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_qwen_final_round_asset_rerun.sh").read_text(encoding="utf-8")

    assert 'DERMAGENT_SCIN_LABEL_SPACE_ID="${DERMAGENT_SCIN_LABEL_SPACE_ID:-scin_grouped}"' in script
    assert 'DERMAGENT_SD198_LABEL_SPACE_ID="${DERMAGENT_SD198_LABEL_SPACE_ID:-sd198_grouped}"' in script
    assert "DERMAGENT_DISABLE_MODEL_WORKFLOW_ROUTING=1" in script
