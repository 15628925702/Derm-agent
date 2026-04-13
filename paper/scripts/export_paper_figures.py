from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAPER_ROOT = Path("/root/DermAgent/paper")
FIG_DIR = PAPER_ROOT / "figures"
SCRIPT_DIR = PAPER_ROOT / "scripts"

# Unified paper palette
COLOR_BASELINE = "#C9D6E3"   # muted light blue-gray
COLOR_AGENT = "#2F5D7E"      # deep blue
COLOR_AGENT_LIGHT = "#89A9C0"
COLOR_WARM_1 = "#E8C9A7"     # muted warm sand
COLOR_WARM_2 = "#D9A679"
COLOR_NEUTRAL_1 = "#D8D8D8"
COLOR_NEUTRAL_2 = "#AEB7C2"
COLOR_GREEN_1 = "#BFD8C1"
COLOR_GREEN_2 = "#7EA37E"


def parse_percent(value: str) -> float:
    match = re.search(r"=\s*([0-9.]+)%", value)
    if not match:
        raise ValueError(f"Cannot parse percent from: {value}")
    return float(match.group(1))


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def zh_metric(metric: str) -> str:
    mapping = {
        "Top-1 accuracy": "Top-1 Accuracy",
        "Top-k hit rate": "Top-k Hit Rate",
        "Malignant recall": "Malignant Recall",
        "Error rate": "Error Rate",
    }
    return mapping.get(metric, metric)


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 240


def export_main_results() -> Path:
    df = load_table(PAPER_ROOT / "table_main_controlled_comparison.csv")
    metrics = [zh_metric(x) for x in df["Metric"].tolist()]
    baseline = [parse_percent(x) for x in df["Direct_Qwen_Baseline"]]
    agent = [parse_percent(x) for x in df["DermAgent"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(metrics))
    width = 0.34
    ax.bar([i - width / 2 for i in x], baseline, width, label="Direct Qwen Baseline", color=COLOR_BASELINE, edgecolor="#4A5A6A", linewidth=0.8)
    ax.bar([i + width / 2 for i in x], agent, width, label="DermAgent", color=COLOR_AGENT, edgecolor="#233645", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Main Controlled Comparison")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    out = FIG_DIR / "figure3_main_results_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_stagewise_results() -> Path:
    df = load_table(PAPER_ROOT / "table_core_ablations_completed.csv")
    metrics = [zh_metric(x) for x in df["Metric"].tolist()]
    cols = [
        ("Direct_Qwen_Baseline", "Direct Qwen"),
        ("Stage1_Basic_Skills_Only", "Stage-1 Basic Skills"),
        ("Skill_Bank_Without_Experience", "Skill Bank w/o Experience"),
        ("Experience_Without_Advanced_Skills", "Experience w/o Advanced Skills"),
    ]
    values = {label: [parse_percent(v) for v in df[col]] for col, label in cols}

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = list(range(len(metrics)))
    width = 0.18
    colors = [COLOR_BASELINE, COLOR_AGENT_LIGHT, COLOR_WARM_1, COLOR_AGENT]
    for idx, (label, color) in enumerate(zip(values.keys(), colors)):
        offset = (idx - 1.5) * width
        ax.bar([i + offset for i in x], values[label], width, label=label, color=color, edgecolor="#4A5A6A", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Stage-wise Contribution of Skills and Experience")
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    out = FIG_DIR / "figure4_stagewise_contributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_subset100_ablation() -> Path:
    df = load_table(PAPER_ROOT / "table_ablation_secondary_subset100.csv")
    metrics = [zh_metric(x) for x in df["Metric"].tolist()]
    cols = [
        ("Direct_Qwen_Baseline", "Direct Qwen"),
        ("Full_Anchor_No_Cognition_Update", "No Cognition Update"),
        ("Full_Without_Abstract_Experiences", "No Abstract Experience"),
        ("Full_Without_Specialist_Skills", "No Specialist Skills"),
        ("Full_With_Rule_Controller", "Rule Controller"),
        ("Full_With_Learned_Controller", "Learned Controller"),
    ]
    values = {label: [parse_percent(v) for v in df[col]] for col, label in cols}

    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = list(range(len(metrics)))
    width = 0.12
    colors = [COLOR_NEUTRAL_1, COLOR_BASELINE, COLOR_WARM_1, COLOR_GREEN_1, COLOR_NEUTRAL_2, COLOR_AGENT]
    for idx, (label, color) in enumerate(zip(values.keys(), colors)):
        offset = (idx - 2.5) * width
        ax.bar([i + offset for i in x], values[label], width, label=label, color=color, edgecolor="#4A5A6A", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Full-System Component Ablation on the 100-case Subset")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    out = FIG_DIR / "figure5_subset100_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_skinvl_results() -> Path:
    df = load_table(PAPER_ROOT / "table_skinvl_candidate48_promptv2_full_dedup.csv")
    metrics = [zh_metric(x) for x in df["Metric"].tolist()]
    baseline = [parse_percent(x) for x in df["Direct_SkinVL_Baseline"]]
    agent = [parse_percent(x) for x in df["Agent_SkinVL_Candidate48_PromptV2"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(metrics))
    width = 0.34
    ax.bar([i - width / 2 for i in x], baseline, width, label="Direct SkinVL Baseline", color=COLOR_BASELINE, edgecolor="#4A5A6A", linewidth=0.8)
    ax.bar([i + width / 2 for i in x], agent, width, label="SkinVL + DermAgent", color=COLOR_AGENT, edgecolor="#233645", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Transfer to a Specialist Dermatology Backbone")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    out = FIG_DIR / "figure6_skinvl_transfer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    outputs = [
        export_main_results(),
        export_stagewise_results(),
        export_subset100_ablation(),
        export_skinvl_results(),
    ]
    print("Exported figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
