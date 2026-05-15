# Final Data Status - 2026-05-15

This file records the state that should be preserved before continuing the
remaining final experiments.

## Final-data location

The curated final outputs are stored outside this git repository at:

- `/data/gh/final-data`

They have also been uploaded to a private Hugging Face dataset repository:

- `https://huggingface.co/datasets/Guaogua/dermagent-final-data-20260515`

Upload verification on 2026-05-15:

- Remote files: `19727` total (`19726` final-data files plus Hugging Face `.gitattributes`)
- Top-level remote entries: `README.md`, `pad`, `scin`, `sd198`
- Upload cache was not included in the repository.

Current completed final datasets:

| Dataset | Status | Directory |
|---|---|---|
| PAD-UFES-20 | Complete | `/data/gh/final-data/pad` |
| SCIN | Complete | `/data/gh/final-data/scin` |
| SD198 grouped | Complete | `/data/gh/final-data/sd198` |

Each completed dataset directory should contain only:

- `direct_6models...`
- `hulumed_agent...`
- `doctor_evidence_package`
- `README.md`

Do not recreate `/data/gh/fanal-data`; the correct directory is
`/data/gh/final-data`.

## Workflow files to preserve

The active HuluMed+agent fusion changes are in:

- `memory/fusion_experience/hulumed_workflow_fusion.py`
- `memory/fusion_experience/workflow_fusion_decision.py`

These include the latest SD198 conservative fusion behavior used for the final
SD198 results.

## Remaining final experiments

### HAM10000

Split size:

- Train: `7011`
- Test: `3004`
- Total: `10015`

Still needed:

- Run 6-model direct on the 30% test split.
- Run HuluMed+agent with 70% train memory construction and 30% test inference.
- Export doctor evidence packages for the agent test split.
- Curate outputs into `/data/gh/final-data/ham10000`.
- Write `/data/gh/final-data/ham10000/README.md` using the PAD/SCIN/SD198 README format.

### ISIC2019

Split size:

- Train: `17732`
- Test: `7599`
- Total: `25331`

The previous ISIC2019 direct run completed, but the case-level direct artifacts
are no longer present locally. Only summary metrics and pipeline logs remain.
The HuluMed+agent run had not reached the test split and its partial bootstrap
artifacts are also no longer present locally.

For a complete final package, rerun:

- 6-model direct on all `7599` test cases.
- HuluMed+agent on `17732` train cases plus `7599` test cases.
- Doctor evidence package export for the `7599` agent test cases.
- Curate outputs into `/data/gh/final-data/isic2019`.
- Write `/data/gh/final-data/isic2019/README.md`.

Previous ISIC2019 direct summary metrics retained for sanity checking:

| Model | Cases | Top1 | Top3 | Malignant recall |
|---|---:|---:|---:|---:|
| dermatollama | 7599 | 52.86% | 73.97% | 19.99% |
| hulumed direct | 7599 | 50.41% | 74.59% | 16.65% |
| qwen | 7599 | 49.48% | 68.57% | 25.06% |
| medgemma | 7599 | 46.30% | 59.32% | 3.77% |
| llama | 7599 | 36.58% | 52.85% | 42.67% |
| skinvl | 7599 | 4.54% | 4.55% | 99.34% |

## Cleanup note

Large duplicated intermediate outputs were removed from `DermAgent/paper_data`,
`hf_uploads`, `final-score`, `/data/gh/fanal-data`, and `/data/gh/中间结果`.
The intended long-lived result location is `/data/gh/final-data`.
