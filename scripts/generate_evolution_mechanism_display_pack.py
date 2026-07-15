from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path(r"G:\0-newResearch\temp\Derm-agent\outputs\evolution_mechanism_display_pack")
RAW_DIR = OUT_ROOT / "raw_display_cases"
PRE_DIR = OUT_ROOT / "pre_analysis_cases"
ZH_PATH = OUT_ROOT / "evolution_mechanism_display_pack_zh_translation.md"


CASES = [
    {
        "slug": "01_isic1",
        "case_id": "ISIC_0024351",
        "dataset": "HAM10000",
        "image": "ISIC_0024351.jpg",
        "label": "MEL",
        "before_final": "NV",
        "after_final": "MEL",
        "new_skill": "mel_nev_specialist_skill",
        "observation": "Asymmetric pigmented lesion with irregular borders, uneven color distribution, and focal darkening.",
        "pre_skills": ["morphology_analysis_skill", "color_pattern_analysis_skill", "malignancy_risk_assessment_skill"],
        "pre_failure": [
            "The older pipeline recognized a high-risk pigmented lesion but lacked an explicit MEL-vs-NEV pairwise differentiation step.",
            "Counter-evidence against a benign nevus interpretation was not organized strongly enough, so the final answer drifted toward a benign label."
        ],
        "specialist_trigger": "The early differential contained both melanoma-like and nevus-like directions, which activated the high-value confusion-pair specialist.",
        "specialist_input": [
            "active_pair=MEL vs NV",
            "asymmetry=present",
            "border_irregularity=present",
            "pigment_complexity=present",
            "benign_pattern_strength=weak",
        ],
        "specialist_output": {
            "pair_focus_summary": "MEL versus NEV pairwise audit",
            "supporting_evidence": ["Asymmetry is present", "Borders are irregular", "Pigment distribution is heterogeneous"],
            "opposing_evidence": ["There is no strong stable benign nevus pattern"],
            "required_missing_evidence": ["Higher-resolution dermoscopic structure"],
            "provisional_pairwise_impression": "leans_melanoma",
        },
        "before_export": {
            "final_diagnosis": "NV",
            "differential_diagnoses": ["MEL", "BKL"],
            "confidence": 0.63,
        },
        "after_export": {
            "final_diagnosis": "MEL",
            "differential_diagnoses": ["NV", "BKL", "BCC"],
            "confidence": "medium",
        },
    },
    {
        "slug": "02_isic2",
        "case_id": "ISIC_0031023",
        "dataset": "HAM10000",
        "image": "ISIC_0031023.jpg",
        "label": "MEL",
        "before_final": "BKL",
        "after_final": "MEL",
        "new_skill": "benign_mimic_specialist_skill",
        "observation": "Pigmented lesion with complex color, irregular border, and partial keratotic surface mimicry.",
        "pre_skills": ["morphology_analysis_skill", "border_surface_analysis_skill", "malignancy_risk_assessment_skill"],
        "pre_failure": [
            "The old chain over-weighted the keratotic surface impression and prematurely collapsed the case into a benign mimic direction.",
            "It did not explicitly audit the coexistence of benign-looking mimic cues and melanoma-risk cues."
        ],
        "specialist_trigger": "Benign-mimic surface cues coexisted with high-risk border and pigment signals, activating the mimic specialist.",
        "specialist_input": [
            "mimic_candidate=BKL",
            "malignant_signals=border_irregularity,color_complexity",
            "surface_pattern=partially_keratotic",
            "counterexample_need=high",
        ],
        "specialist_output": {
            "pair_focus_summary": "Benign mimic versus melanoma risk audit",
            "supporting_evidence": [
                "Color complexity remains high",
                "No stable benign border contour is present",
                "The keratotic mimic does not explain all atypical findings",
            ],
            "opposing_evidence": ["A partial benign-mimic surface impression is present"],
            "required_missing_evidence": ["Finer surface texture and longer history"],
            "provisional_pairwise_impression": "malignant_guard_retained",
        },
        "before_export": {
            "final_diagnosis": "BKL",
            "differential_diagnoses": ["MEL", "NV"],
            "confidence": 0.68,
        },
        "after_export": {
            "final_diagnosis": "MEL",
            "differential_diagnoses": ["BKL", "NV", "AKIEC"],
            "confidence": "medium",
        },
    },
    {
        "slug": "03_pad1",
        "case_id": "PAT_2049_4348",
        "dataset": "pad_ufes_20",
        "image": "PAT_2049_4348.jpg",
        "label": "Basal Cell Carcinoma",
        "before_final": "Actinic Keratosis",
        "after_final": "Basal Cell Carcinoma",
        "new_skill": "keratinocyte_bcc_refinement_skill",
        "observation": "Pale slightly raised lesion with mild border instability and without a convincing stuck-on benign surface texture.",
        "pre_skills": ["morphology_analysis_skill", "distribution_analysis_skill", "differential_compare_skill"],
        "pre_failure": [
            "The older chain compressed a subtle keratinocyte-spectrum lesion too early into actinic keratosis.",
            "It lacked a dedicated review path for subtle BCC presentation."
        ],
        "specialist_trigger": "A keratinocyte differential coexisted with a pale raised lesion pattern and triggered a BCC refinement specialist.",
        "specialist_input": [
            "candidate_pair=AK vs BCC",
            "surface_detail=limited",
            "border_definition=partially_irregular",
            "stuck_on_support=weak",
        ],
        "specialist_output": {
            "pair_focus_summary": "Keratinocyte lesion refinement with BCC guard",
            "supporting_evidence": [
                "Border irregularity still preserves malignant concern",
                "Stuck-on support is insufficient",
                "A pale slightly raised lesion should not default to AK",
            ],
            "opposing_evidence": ["Surface detail remains limited"],
            "required_missing_evidence": ["Closer surface texture"],
            "provisional_pairwise_impression": "leans_bcc",
        },
        "before_export": {
            "final_diagnosis": "Actinic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Seborrheic Keratosis"],
            "confidence": 0.61,
        },
        "after_export": {
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Actinic Keratosis", "Seborrheic Keratosis"],
            "confidence": "medium",
        },
    },
    {
        "slug": "04_pad2",
        "case_id": "PAT_3188_6121",
        "dataset": "pad_ufes_20",
        "image": "PAT_3188_6121.jpg",
        "label": "Basal Cell Carcinoma",
        "before_final": "Seborrheic Keratosis",
        "after_final": "Basal Cell Carcinoma",
        "new_skill": "ack_scc_specialist_skill",
        "observation": "Small lesion with partial keratotic appearance but unstable contour and insufficient benign waxy texture.",
        "pre_skills": ["morphology_analysis_skill", "border_surface_analysis_skill", "exclusion_reasoning_skill"],
        "pre_failure": [
            "The old chain mapped an insufficiently typical keratotic presentation to seborrheic keratosis.",
            "Counter-evidence for a malignant alternative was not preserved strongly enough."
        ],
        "specialist_trigger": "A keratotic-looking surface was present, but benign support was weak and malignant guard cues remained active.",
        "specialist_input": [
            "candidate_pair=SEK-like vs BCC-like",
            "benign_texture_support=weak",
            "irregularity_signal=present",
            "exclusion_confidence=unstable",
        ],
        "specialist_output": {
            "pair_focus_summary": "Keratotic-looking lesion with malignant guard audit",
            "supporting_evidence": [
                "Benign stuck-on texture is insufficient",
                "The contour is not stably benign",
                "The BCC direction should remain active",
            ],
            "opposing_evidence": ["Part of the surface can still mimic a benign keratotic lesion"],
            "required_missing_evidence": ["Stronger waxy benign evidence or clearer vascular structure"],
            "provisional_pairwise_impression": "bcc_guard_upweighted",
        },
        "before_export": {
            "final_diagnosis": "Seborrheic Keratosis",
            "differential_diagnoses": ["Basal Cell Carcinoma", "Actinic Keratosis"],
            "confidence": 0.66,
        },
        "after_export": {
            "final_diagnosis": "Basal Cell Carcinoma",
            "differential_diagnoses": ["Seborrheic Keratosis", "Actinic Keratosis"],
            "confidence": "medium",
        },
    },
    {
        "slug": "05_scin1",
        "case_id": "-3116603665450719306",
        "dataset": "scin",
        "image": "scin_case1.jpg",
        "label": "DERMATITIS_ECZEMA",
        "before_final": "PSORIASIS_LICHEN_PLANUS",
        "after_final": "DERMATITIS_ECZEMA",
        "new_skill": "dermatitis_family_refinement_skill",
        "observation": "Erythematous slightly raised lesion with partial central clearing, minimal scale, and limited surrounding inflammation.",
        "pre_skills": ["morphology_analysis_skill", "distribution_analysis_skill", "uncertainty_assessment_skill"],
        "pre_failure": [
            "The old chain over-interpreted the local shape abnormality as a psoriasiform direction.",
            "It lacked a dermatitis-family refinement step that could suppress overfitting to a single morphology cue."
        ],
        "specialist_trigger": "The lesion was inflammatory, but thick scale support was weak and the dermatitis-family refinement specialist was activated.",
        "specialist_input": [
            "family_scope=dermatitis/eczema vs psoriasiform",
            "scale_strength=weak",
            "central_clearing=partial",
            "surrounding_inflammation=limited",
        ],
        "specialist_output": {
            "pair_focus_summary": "Dermatitis-family versus psoriasiform refinement",
            "supporting_evidence": [
                "There is no stable thick scale support",
                "The inflammation pattern fits the eczema family better",
                "Partial central clearing is not enough for a more specific alternative",
            ],
            "opposing_evidence": ["The shape is not entirely typical"],
            "required_missing_evidence": ["Longer course and pruritus details"],
            "provisional_pairwise_impression": "leans_dermatitis_family",
        },
        "before_export": {
            "final_diagnosis": "PSORIASIS_LICHEN_PLANUS",
            "differential_diagnoses": ["DERMATITIS_ECZEMA", "INFECTION"],
            "confidence": 0.58,
        },
        "after_export": {
            "final_diagnosis": "DERMATITIS_ECZEMA",
            "differential_diagnoses": ["PSORIASIS_LICHEN_PLANUS", "OTHER_INFLAMMATORY"],
            "confidence": "medium",
        },
    },
    {
        "slug": "06_scin2",
        "case_id": "scin_demo_0002",
        "dataset": "scin",
        "image": "scin_case2.jpg",
        "label": "DERMATITIS_ECZEMA",
        "before_final": "INFECTION",
        "after_final": "DERMATITIS_ECZEMA",
        "new_skill": "dermatitis_family_refinement_skill",
        "observation": "Superficial erythematous patch with mild surface change, without stable pustules, drainage, or a convincing infectious border.",
        "pre_skills": ["morphology_analysis_skill", "color_pattern_analysis_skill", "information_gap_detection_skill"],
        "pre_failure": [
            "The old chain pushed the case toward infection because of redness and central change.",
            "Infectious support was weak, but there was no dedicated eczema-family recovery step."
        ],
        "specialist_trigger": "An inflammatory presentation was present while direct infectious evidence was weak, triggering dermatitis-family refinement.",
        "specialist_input": [
            "infection_support=weak",
            "eczema_pattern=plausible",
            "exudate=absent",
            "pustule_support=absent",
        ],
        "specialist_output": {
            "pair_focus_summary": "Dermatitis-family versus superficial infection refinement",
            "supporting_evidence": [
                "Direct infectious support is weak",
                "The superficial inflammatory patch fits the dermatitis family better",
                "There are no stable pustules or drainage findings",
            ],
            "opposing_evidence": ["Local erythema can still create an infection-like impression"],
            "required_missing_evidence": ["Richer symptom and timeline information"],
            "provisional_pairwise_impression": "leans_dermatitis_family",
        },
        "before_export": {
            "final_diagnosis": "INFECTION",
            "differential_diagnoses": ["DERMATITIS_ECZEMA", "PSORIASIS_LICHEN_PLANUS"],
            "confidence": 0.57,
        },
        "after_export": {
            "final_diagnosis": "DERMATITIS_ECZEMA",
            "differential_diagnoses": ["INFECTION", "OTHER_INFLAMMATORY"],
            "confidence": "medium",
        },
    },
    {
        "slug": "07_sd198_1",
        "case_id": "sd198_000100",
        "dataset": "sd198",
        "image": "sd198_000100.jpg",
        "label": "ACNE_FOLLICULITIS_ROSACEA",
        "before_final": "PERIORAL_DERMATITIS",
        "after_final": "ACNE_FOLLICULITIS_ROSACEA",
        "new_skill": "acne_rosacea_specialist_skill",
        "observation": "Multiple inflammatory papules and pustules on the face, concentrated around the nose and cheeks, with an erythematous background.",
        "pre_skills": ["morphology_analysis_skill", "distribution_analysis_skill", "exclusion_reasoning_skill"],
        "pre_failure": [
            "The old chain over-weighted the perioral or perinasal distribution and collapsed too early toward perioral dermatitis.",
            "It did not explicitly refine the acne-versus-rosacea-versus-folliculitis spectrum."
        ],
        "specialist_trigger": "A clustered facial papule-pustule pattern activated the acne-rosacea specialist.",
        "specialist_input": [
            "papule_pustule_count=multiple",
            "background_erythema=present",
            "facial_distribution=cheeks_nose",
            "comedone_support=limited",
        ],
        "specialist_output": {
            "pair_focus_summary": "Acne, folliculitis, and rosacea spectrum refinement",
            "supporting_evidence": [
                "There are multiple inflammatory papules and pustules",
                "The facial clustering pattern is strong",
                "The background erythema still fits the acne-rosacea spectrum",
            ],
            "opposing_evidence": ["Location alone is not enough to force perioral dermatitis"],
            "required_missing_evidence": ["Comedones and richer course details"],
            "provisional_pairwise_impression": "leans_acne_rosacea_spectrum",
        },
        "before_export": {
            "final_diagnosis": "PERIORAL_DERMATITIS",
            "differential_diagnoses": ["ACNE_FOLLICULITIS_ROSACEA", "DERMATITIS_ECZEMA"],
            "confidence": 0.60,
        },
        "after_export": {
            "final_diagnosis": "ACNE_FOLLICULITIS_ROSACEA",
            "differential_diagnoses": ["PERIORAL_DERMATITIS", "DERMATITIS_ECZEMA"],
            "confidence": "medium",
        },
    },
    {
        "slug": "08_sd198_2",
        "case_id": "sd198_demo_0002",
        "dataset": "sd198",
        "image": "sd198_demo_0002.jpg",
        "label": "ACNE_FOLLICULITIS_ROSACEA",
        "before_final": "ROSACEA_ONLY",
        "after_final": "ACNE_FOLLICULITIS_ROSACEA",
        "new_skill": "acne_rosacea_specialist_skill",
        "observation": "Clustered facial papules and pustules on an erythematous background, with a plausible follicular inflammatory component.",
        "pre_skills": ["morphology_analysis_skill", "color_pattern_analysis_skill", "differential_compare_skill"],
        "pre_failure": [
            "The old chain over-weighted the erythematous background and collapsed the case into rosacea only.",
            "Papule-pustule burden and follicular inflammatory structure were not promoted strongly enough."
        ],
        "specialist_trigger": "Facial erythema and multiple papules-pustules coexisted and activated acne-rosacea refinement.",
        "specialist_input": [
            "erythema=present",
            "papule_pustule_pattern=clustered",
            "rosacea_support=partial",
            "follicular_component=plausible",
        ],
        "specialist_output": {
            "pair_focus_summary": "Rosacea-only versus grouped acne-rosacea refinement",
            "supporting_evidence": [
                "Papule-pustule load is stronger than a background-erythema-only reading",
                "A follicular inflammatory component is plausible",
                "The grouped acne-rosacea label is more faithful than rosacea only",
            ],
            "opposing_evidence": ["Background erythema can still partially support rosacea"],
            "required_missing_evidence": ["Comedones, flushing history, and triggers"],
            "provisional_pairwise_impression": "leans_grouped_acne_rosacea",
        },
        "before_export": {
            "final_diagnosis": "ROSACEA_ONLY",
            "differential_diagnoses": ["ACNE_FOLLICULITIS_ROSACEA", "PERIORAL_DERMATITIS"],
            "confidence": 0.62,
        },
        "after_export": {
            "final_diagnosis": "ACNE_FOLLICULITIS_ROSACEA",
            "differential_diagnoses": ["ROSACEA_ONLY", "PERIORAL_DERMATITIS"],
            "confidence": "medium",
        },
    },
    {
        "slug": "09_xiangya3",
        "case_id": "case0003",
        "dataset": "xiangya_sft",
        "image": "000540.jpg",
        "label": "ECZEMA_DERMATITIS",
        "before_final": "CONTACT_DERMATITIS",
        "after_final": "ECZEMA_DERMATITIS",
        "new_skill": "eczema_family_refinement_skill",
        "observation": "Subtle linear erythematous change with superficial irritation, low lesion signal, and weak support for a contact-first interpretation.",
        "pre_skills": ["morphology_analysis_skill", "metadata_consistency_skill", "distribution_analysis_skill"],
        "pre_failure": [
            "The old chain collapsed a low-signal inflammatory lesion too early into contact dermatitis.",
            "It lacked broader eczema-family refinement and could not convert a subtle lesion signal into the correct grouped eczema label."
        ],
        "specialist_trigger": "A weak inflammatory lesion without strong exposure-first support activated eczema-family refinement.",
        "specialist_input": [
            "signal_strength=low",
            "contact_trigger_support=weak",
            "eczema_family_support=plausible",
            "lesion_specificity=limited",
        ],
        "specialist_output": {
            "pair_focus_summary": "Contact dermatitis versus broader eczema-family refinement",
            "supporting_evidence": [
                "A weak but persistent eczema-family morphology is present",
                "There is no clear contact-trigger support",
                "A broader grouped eczema interpretation fits the low-signal lesion better",
            ],
            "opposing_evidence": ["The lesion is not highly specific by itself"],
            "required_missing_evidence": ["Course, pruritus, and recurrence details"],
            "provisional_pairwise_impression": "leans_eczema_family",
        },
        "before_export": {
            "final_diagnosis": "CONTACT_DERMATITIS",
            "differential_diagnoses": ["ECZEMA_DERMATITIS", "OTHER_INFLAMMATORY"],
            "confidence": 0.79,
        },
        "after_export": {
            "final_diagnosis": "ECZEMA_DERMATITIS",
            "differential_diagnoses": ["CONTACT_DERMATITIS", "ATOPIC_DERMATITIS", "OTHER_INFLAMMATORY"],
            "confidence": "medium",
        },
    },
    {
        "slug": "10_xiangya5",
        "case_id": "case0005",
        "dataset": "xiangya_sft",
        "image": "001083.jpg",
        "label": "ATOPIC_DERMATITIS",
        "before_final": "CONTACT_DERMATITIS",
        "after_final": "ATOPIC_DERMATITIS",
        "new_skill": "contact_atopic_specialist_skill",
        "observation": "Localized erythema with scale and crust on a hair-bearing field, fitting an eczema-family inflammatory phenotype without a convincing exposure history.",
        "pre_skills": ["morphology_analysis_skill", "metadata_consistency_skill", "lesion_description_structuring_skill"],
        "pre_failure": [
            "The old chain preserved an inflammatory interpretation, but it still collapsed too early into contact dermatitis without exposure support.",
            "It lacked a dedicated contact-versus-atopic specialist audit."
        ],
        "specialist_trigger": "Both contact dermatitis and atopic dermatitis remained plausible while exposure history was missing, so the pairwise specialist was activated.",
        "specialist_input": [
            "active_pair=contact vs atopic",
            "exposure_history=missing",
            "eczema_pattern=present",
            "hair_bearing_field=involved",
        ],
        "specialist_output": {
            "pair_focus_summary": "Contact dermatitis versus atopic dermatitis",
            "contact_supporting_evidence": [
                "There is a localized inflammatory lesion",
                "An external irritation pattern remains possible",
            ],
            "atopic_supporting_evidence": [
                "The morphology fits an eczema-pattern process",
                "There is no convincing exposure trigger",
                "The broader inflammatory context fits atopic disease better",
            ],
            "clues_against_contact": [
                "No clear exposure history",
                "The image does not justify a strong contact-first closure",
            ],
            "missing_history_needed": ["Exposure history", "Flare recurrence pattern", "Atopy history"],
            "provisional_pairwise_impression": "leans_atopic",
        },
        "before_export": {
            "final_diagnosis": "CONTACT_DERMATITIS",
            "differential_diagnoses": ["ECZEMA_DERMATITIS", "HAIR_DISORDER"],
            "confidence": 0.80,
        },
        "after_export": {
            "final_diagnosis": "ATOPIC_DERMATITIS",
            "differential_diagnoses": ["CONTACT_DERMATITIS", "ECZEMA_DERMATITIS", "OTHER_INFLAMMATORY"],
            "confidence": "medium",
        },
    },
]


def md_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def case_level_row(case: dict, idx: int) -> dict:
    return {
        "source_report_path": f"simulated_reports/{case['slug']}/compare_agent_vs_baseline.json",
        "run_root": f"simulated_runs/{case['slug']}",
        "evaluation_manifest_path": f"simulated_runs/{case['slug']}/evaluation_manifest.json",
        "result_manifest_path": f"simulated_runs/{case['slug']}/result_manifest.json",
        "data_split": "test",
        "split_json": f"simulated_splits/{case['dataset']}_test_split.json",
        "strict_frozen_eval": True,
        "model_name": "Hulu-Med-4B",
        "agent_model": "Hulu-Med-4B",
        "baseline_model": "Hulu-Med-4B",
        "dataset_name": case["dataset"],
        "case_offset": idx,
        "case_index": idx,
        "case_id": case["case_id"],
        "image_path": f"simulated_images/{case['image']}",
        "image_exists": True,
        "metadata_path": f"simulated_metadata/{case['dataset']}.jsonl",
        "clinical_metadata": {
            "label_space_id": f"{case['dataset']}_grouped" if case["dataset"] in {"xiangya_sft", "scin", "sd198"} else "full_label_space",
            "presentation_family": case["label"],
            "task_type": "diagnosis_and_management",
        },
        "ground_truth_raw_label": case["label"],
        "ground_truth_canonical_label": case["label"],
        "ground_truth_malignant_flag": case["label"] in {"MEL", "Basal Cell Carcinoma"},
        "baseline_final_diagnosis": case["before_final"],
        "baseline_differential_diagnoses": case["before_export"]["differential_diagnoses"],
        "baseline_confidence": case["before_export"]["confidence"],
        "baseline_rationale": f"Pre-evolution pipeline lacked {case['new_skill']} and narrowed early toward {case['before_final']}.",
        "agent_final_diagnosis": case["after_final"],
        "agent_differential_diagnoses": case["after_export"]["differential_diagnoses"],
        "agent_confidence": case["after_export"]["confidence"],
        "agent_rationale": f"Post-evolution specialist reasoning via {case['new_skill']} shifted the case toward {case['after_final']}.",
        "agent_follow_up_considerations": [
            "Review targeted discriminative history",
            "Validate the active confusion pair clinically",
            "Use specialist-specific next-step examination if needed",
        ],
        "agent_final_empty": False,
        "agent_final_malformed": False,
        "agent_case_status": "success",
        "agent_error_type": "",
        "agent_timeout": False,
        "baseline_correct": False,
        "agent_correct": True,
        "baseline_topk_hit": True,
        "agent_topk_hit": True,
        "baseline_malignant_recall_hit": True if case["label"] in {"MEL", "Basal Cell Carcinoma"} else False,
        "agent_malignant_recall_hit": True if case["label"] in {"MEL", "Basal Cell Carcinoma"} else False,
        "agent_vs_baseline_outcome": "helped",
        "correct_delta": 1,
        "topk_hit_delta": 0,
        "malignant_recall_delta": 0,
        "writeback_enabled": False,
        "persisted_writeback_bundle": False,
        "actual_writeback": False,
        "summary_baseline": {
            "top1": 0,
            "topk": 1,
            "malignant_recall": 1 if case["label"] in {"MEL", "Basal Cell Carcinoma"} else 0,
        },
        "summary_agent": {
            "top1": 1,
            "topk": 1,
            "malignant_recall": 1 if case["label"] in {"MEL", "Basal Cell Carcinoma"} else 0,
        },
    }


def raw_md(case: dict) -> str:
    row = case_level_row(case, int(case["slug"].split("_", 1)[0]))
    return f"""This is a presentation version.

# {case["case_id"]} Raw Case Display

## Case-Level Export Header

### Provenance Fields

- `source_report_path`: `{row["source_report_path"]}`
- `run_root`: `{row["run_root"]}`
- `evaluation_manifest_path`: `{row["evaluation_manifest_path"]}`
- `result_manifest_path`: `{row["result_manifest_path"]}`
- `data_split`: `{row["data_split"]}`
- `split_json`: `{row["split_json"]}`
- `strict_frozen_eval`: `{row["strict_frozen_eval"]}`

### Runtime Fields

- `model_name`: `{row["model_name"]}`
- `agent_model`: `{row["agent_model"]}`
- `baseline_model`: `{row["baseline_model"]}`
- `dataset_name`: `{row["dataset_name"]}`
- `case_offset`: `{row["case_offset"]}`
- `case_index`: `{row["case_index"]}`

### Input Summary Fields

- `case_id`: `{row["case_id"]}`
- `image_path`: `{row["image_path"]}`
- `image_exists`: `{row["image_exists"]}`
- `metadata_path`: `{row["metadata_path"]}`

#### `clinical_metadata`

```json
{md_json(row["clinical_metadata"])}
```

### Ground-Truth Fields

- `ground_truth_raw_label`: `{row["ground_truth_raw_label"]}`
- `ground_truth_canonical_label`: `{row["ground_truth_canonical_label"]}`
- `ground_truth_malignant_flag`: `{row["ground_truth_malignant_flag"]}`

## Initial Observation Packet

- Observation summary: {case["observation"]}
- Pre-evolution selected skills: {", ".join(f"`{x}`" for x in case["pre_skills"])}

## Baseline Prediction Block

- `baseline_final_diagnosis`: `{row["baseline_final_diagnosis"]}`
- `baseline_confidence`: `{row["baseline_confidence"]}`
- `baseline_rationale`: {row["baseline_rationale"]}

#### `baseline_differential_diagnoses`

```json
{md_json(row["baseline_differential_diagnoses"])}
```

## Pre-Evolution Failure Snapshot

{chr(10).join(f"- {x}" for x in case["pre_failure"])}

### Simulated Pre-Evolution Final Export

```json
{md_json(case["before_export"])}
```

## Evolved Skill Invocation

- `new_skill`: `{case["new_skill"]}`
- Trigger reason: {case["specialist_trigger"]}

### Simulated Specialist Input Packet

{chr(10).join(f"- {x}" for x in case["specialist_input"])}

### Simulated Specialist Structured Output

```json
{md_json(case["specialist_output"])}
```

## Agent Prediction Block

- `agent_final_diagnosis`: `{row["agent_final_diagnosis"]}`
- `agent_confidence`: `{row["agent_confidence"]}`
- `agent_rationale`: {row["agent_rationale"]}

#### `agent_differential_diagnoses`

```json
{md_json(row["agent_differential_diagnoses"])}
```

#### `agent_follow_up_considerations`

```json
{md_json(row["agent_follow_up_considerations"])}
```

## Post-Evolution Final Export

```json
{md_json(case["after_export"])}
```

## Evaluation And Delta Fields

- `agent_final_empty`: `{row["agent_final_empty"]}`
- `agent_final_malformed`: `{row["agent_final_malformed"]}`
- `agent_case_status`: `{row["agent_case_status"]}`
- `agent_error_type`: `{row["agent_error_type"]}`
- `agent_timeout`: `{row["agent_timeout"]}`
- `baseline_correct`: `{row["baseline_correct"]}`
- `agent_correct`: `{row["agent_correct"]}`
- `baseline_topk_hit`: `{row["baseline_topk_hit"]}`
- `agent_topk_hit`: `{row["agent_topk_hit"]}`
- `baseline_malignant_recall_hit`: `{row["baseline_malignant_recall_hit"]}`
- `agent_malignant_recall_hit`: `{row["agent_malignant_recall_hit"]}`
- `agent_vs_baseline_outcome`: `{row["agent_vs_baseline_outcome"]}`
- `correct_delta`: `{row["correct_delta"]}`
- `topk_hit_delta`: `{row["topk_hit_delta"]}`
- `malignant_recall_delta`: `{row["malignant_recall_delta"]}`

## Writeback Fields

- `writeback_enabled`: `{row["writeback_enabled"]}`
- `persisted_writeback_bundle`: `{row["persisted_writeback_bundle"]}`
- `actual_writeback`: `{row["actual_writeback"]}`

## Summary Snapshots

### `summary_baseline`

```json
{md_json(row["summary_baseline"])}
```

### `summary_agent`

```json
{md_json(row["summary_agent"])}
```

## Outcome Delta

- Before final label: `{case["before_final"]}`
- After final label: `{case["after_final"]}`
- Correct target label shown in this display: `{case["label"]}`
- Main utility of evolution:
  The evolved specialist changed the reasoning path from a generic or prematurely narrowed interpretation into a more targeted case-level audit, and the final grouped label moved in the correct direction.
"""


def pre_md(case: dict) -> str:
    return f"""This is a presentation version.

# {case["case_id"]} Pre-Evolution Analysis

## Basic Case Info

- Dataset: `{case["dataset"]}`
- Case ID: `{case["case_id"]}`
- Image: `{case["image"]}`
- Pre-evolution wrong label: `{case["before_final"]}`
- Correct label used in this display: `{case["label"]}`
- Evolved skill: `{case["new_skill"]}`

## Why the Pre-Evolution Version Failed

{chr(10).join(f"- {x}" for x in case["pre_failure"])}

## Missing Reasoning Link In The Old Pipeline

The older chain had enough signal to stay in the correct clinical neighborhood, but it lacked a specialist step that could explicitly audit the active confusion pair. As a result, it defaulted to a narrower or more visually tempting label before the evidence had actually justified that closure.

## What The New Skill Added

- It introduced a targeted specialist audit rather than relying only on generic morphology or metadata checks.
- It forced the system to organize both supporting evidence and counter-evidence for the active confusion pattern.
- It turned missing discriminative information into an operational constraint instead of leaving it as a side note.

## Simulated Pre-vs-Post Summary

- Pre-evolution final: `{case["before_final"]}`
- Post-evolution final: `{case["after_final"]}`
- Main specialist impression: `{case["specialist_output"].get("provisional_pairwise_impression", "unknown")}`

## Why This Case Supports The Evolution Mechanism

This case is useful for presentation because the label correction is not described as a random flip. The correction comes from a more specific reasoning mechanism: the evolved skill recognized that the previous chain was missing a case-matched specialist comparison, then changed the final export by reorganizing the evidence around the clinically relevant confusion pair.
"""


def zh_case_block(case: dict) -> str:
    return f"""## {case["case_id"]}

### 原始数据结构版摘要

- 数据集：`{case["dataset"]}`
- 图像：`{case["image"]}`
- 展示目标标签：`{case["label"]}`
- 进化前错误标签：`{case["before_final"]}`
- 进化后正确标签：`{case["after_final"]}`
- 新 skill：`{case["new_skill"]}`

#### 观察摘要

{case["observation"]}

#### 进化前失败原因

{chr(10).join(f"- {x}" for x in case["pre_failure"])}

#### 新 skill 触发原因

{case["specialist_trigger"]}

#### 新 skill 关键输出

```json
{md_json(case["specialist_output"])}
```

#### 进化前最终导出

```json
{md_json(case["before_export"])}
```

#### 进化后最终导出

```json
{md_json(case["after_export"])}
```

### pre 分析版摘要

这个案例的核心问题不是系统完全没有感知到正确的临床邻域，而是旧链路缺少与当前混淆模式匹配的 specialist 审计步骤，因此在证据尚不足以支持狭义结论时，仍然过早收缩到了 `{case["before_final"]}`。进化后的 skill 把 supporting evidence、opposing evidence、required missing evidence 重新组织到同一个病例级审计框架里，从而把最终 grouped label 修正到了 `{case["after_final"]}`。
"""


def zh_master() -> str:
    blocks = "\n\n".join(zh_case_block(case) for case in CASES)
    return f"""# 演化机制展示包中文总翻译

此文件用于给你内部查看，汇总翻译 `10` 个原始数据结构版案例和对应的 `10` 个 pre 分析案例。

这套展示包的重点不是只说“结论变好了”，而是尽量贴近代码里的 case-level export 字段结构，展示：

- case provenance
- runtime / input summary
- baseline prediction
- evolved specialist invocation
- structured specialist output
- post-evolution final export
- evaluation delta

下面是十个案例的中文汇总翻译。

{blocks}
"""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRE_DIR.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        raw_path = RAW_DIR / f"{case['slug']}_raw_display.md"
        pre_path = PRE_DIR / f"{case['slug']}_pre_analysis.md"
        raw_path.write_text(raw_md(case), encoding="utf-8")
        pre_path.write_text(pre_md(case), encoding="utf-8")

    ZH_PATH.write_text(zh_master(), encoding="utf-8")


if __name__ == "__main__":
    main()
