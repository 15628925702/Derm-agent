from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact same-case before/after evolution demo.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--baseline-record", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Hulu-Med-4B")
    parser.add_argument("--observation-json", default="", help="Optional precomputed observation JSON string.")
    parser.add_argument("--observation-file", default="", help="Optional path to precomputed observation JSON.")
    return parser.parse_args()


def call_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False),
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def build_image_message(prompt: str, image_path: str) -> dict[str, Any]:
    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }


def synthesize_after_final(
    *,
    baseline_final: dict[str, Any],
    observation: dict[str, Any],
    after_specialist: dict[str, Any],
) -> dict[str, Any]:
    impression = str(after_specialist.get("provisional_pairwise_impression", "")).strip().lower()
    if impression == "leans_atopic":
        final_label = "ATOPIC_DERMATITIS"
        confidence = "medium"
    elif impression == "indeterminate":
        final_label = "ECZEMA_DERMATITIS"
        confidence = "low"
    else:
        final_label = str(baseline_final.get("final_diagnosis", "CONTACT_DERMATITIS") or "CONTACT_DERMATITIS")
        confidence = "low"

    rationale_bits = [
        f"Baseline preview favored {baseline_final.get('final_diagnosis', 'CONTACT_DERMATITIS')}.",
        f"Post-evolution specialist impression was {impression or 'unknown'}.",
    ]
    clues_against_contact = [str(item).strip() for item in after_specialist.get("clues_against_contact", []) if str(item).strip()]
    missing_history = [str(item).strip() for item in after_specialist.get("missing_history_needed", []) if str(item).strip()]
    if clues_against_contact:
        rationale_bits.append("Clues against narrow contact dermatitis included " + "; ".join(clues_against_contact[:2]) + ".")
    if missing_history:
        rationale_bits.append("Key missing history included " + "; ".join(missing_history[:2]) + ".")

    differential = [final_label]
    if final_label != "CONTACT_DERMATITIS":
        differential.append("CONTACT_DERMATITIS")
    if "ATOPIC_DERMATITIS" not in differential:
        differential.append("ATOPIC_DERMATITIS")
    return {
        "final_diagnosis": final_label,
        "differential_diagnoses": differential[:3],
        "rationale": " ".join(rationale_bits),
        "confidence": confidence,
        "follow_up_considerations": [
            "Clarify exposure history and chronicity before committing to a narrow contact-dermatitis label.",
            "Correlate with itch pattern, recurrence history, and broader eczematous distribution.",
        ],
        "_generation_mode": "heuristic_fallback_due_final_prompt_instability",
        "_observation_summary": observation.get("image_summary", ""),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_partial(name: str, payload: Any) -> None:
        (output_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_record = json.loads(Path(args.baseline_record).read_text(encoding="utf-8"))
    baseline_final = baseline_record.get("qwen_final", {})
    case_id = str(baseline_record.get("case_id", "case0005"))
    metadata = baseline_record.get("input_summary", {}).get("clinical_metadata", {})
    label_space = baseline_record.get("input_summary", {}).get("label_space", {}).get("canonical_labels", [])
    metadata_summary = (
        f"dominant_family={metadata.get('dominant_family', 'unknown')}; "
        f"related_category={metadata.get('related_category', 'unknown')}; "
        f"task_type={metadata.get('task_type', 'unknown')}; "
        "exposure_history=missing"
    )

    if args.observation_file.strip():
        observation = json.loads(Path(args.observation_file).read_text(encoding="utf-8-sig"))
    elif args.observation_json.strip():
        observation = json.loads(args.observation_json)
    else:
        observation = call_json(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=96,
            messages=[
                {
                    "role": "system",
                    "content": "You are a dermatology observation assistant. Return JSON only.",
                },
                build_image_message(
                    "Summarize the visible dermatologic findings only. "
                    "Return JSON with keys: image_summary, key_clues, uncertainty.",
                    args.image_path,
                ),
            ],
        )
    save_partial("observation", observation)

    before_evidence = call_json(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=160,
        messages=[
            {
                "role": "system",
                "content": "You are a compact DermAgent pre-evolution evidence packer. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Build a minimal pre-evolution evidence summary using only old generic skills.\n"
                    "Return JSON with keys: morphology_summary, metadata_consistency_summary, uncertainty_summary.\n"
                    f"Observation: {observation.get('image_summary', '')}. "
                    f"DDx: {', '.join(observation.get('ddx_candidates', []))}. "
                    f"Metadata: {metadata_summary}"
                ),
            },
        ],
    )
    save_partial("before_evidence", before_evidence)

    before_final = call_json(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=160,
        messages=[
            {
                "role": "system",
                "content": "You are the compact pre-evolution final diagnosis exporter. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Return JSON with keys: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                    "This is the pre-evolution path with no new specialist skill.\n"
                    f"Allowed labels: {', '.join(label_space)}. "
                    f"Baseline preview: {baseline_final.get('final_diagnosis', '')}. "
                    f"Observation: {observation.get('image_summary', '')}. "
                    f"Old-skill evidence: {json.dumps(before_evidence, ensure_ascii=False)}"
                ),
            },
        ],
    )
    save_partial("before_final", before_final)

    after_specialist = call_json(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=192,
        messages=[
            {
                "role": "system",
                "content": "You are the new contact-vs-atopic specialist skill added after evolution. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Focus only on contact dermatitis versus atopic dermatitis.\n"
                    "Return JSON with keys: pair_focus_summary, contact_supporting_evidence, atopic_supporting_evidence, "
                    "clues_against_contact, missing_history_needed, provisional_pairwise_impression.\n"
                    "If evidence is weak, say the pair is underdetermined.\n"
                    f"Observation: {observation.get('image_summary', '')}. "
                    f"Notes: {', '.join(observation.get('notes', []))}. "
                    f"Metadata: {metadata_summary}"
                ),
            },
        ],
    )
    save_partial("after_specialist", after_specialist)

    try:
        after_final = call_json(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=192,
            messages=[
                {
                    "role": "system",
                    "content": "You are the compact post-evolution final diagnosis exporter. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON with keys: final_diagnosis, differential_diagnoses, rationale, confidence, follow_up_considerations.\n"
                        "This is the post-evolution path. Use the new specialist output to avoid over-narrowing when evidence is weak.\n"
                        f"Allowed labels: {', '.join(label_space)}. "
                        f"Baseline preview: {baseline_final.get('final_diagnosis', '')}. "
                        f"Observation: {observation.get('image_summary', '')}. "
                        f"Specialist impression: {after_specialist.get('provisional_pairwise_impression', 'unknown')}. "
                        f"Against contact: {', '.join(after_specialist.get('clues_against_contact', [])[:2])}. "
                        f"Missing history: {', '.join(after_specialist.get('missing_history_needed', [])[:2])}. "
                        "If the pair is underdetermined, prefer a broader or more cautious grouped label rather than a narrow contact diagnosis."
                    ),
                },
            ],
        )
    except requests.HTTPError:
        after_final = synthesize_after_final(
            baseline_final=baseline_final,
            observation=observation,
            after_specialist=after_specialist,
        )
    save_partial("after_final", after_final)

    result = {
        "case_id": case_id,
        "image_path": args.image_path,
        "observation": observation,
        "before": {
            "evidence": before_evidence,
            "final_export": before_final,
        },
        "after": {
            "specialist_skill": after_specialist,
            "final_export": after_final,
        },
    }

    (output_dir / "same_case_evolution_demo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = [
        f"# Same-Case Evolution Demo: {case_id}",
        "",
        "## Observation",
        "",
        json.dumps(observation, ensure_ascii=False, indent=2),
        "",
        "## Before Evolution",
        "",
        "### Old Skill Evidence",
        "",
        json.dumps(before_evidence, ensure_ascii=False, indent=2),
        "",
        "### Final Export",
        "",
        json.dumps(before_final, ensure_ascii=False, indent=2),
        "",
        "## After Evolution",
        "",
        "### New Specialist Skill",
        "",
        json.dumps(after_specialist, ensure_ascii=False, indent=2),
        "",
        "### Final Export",
        "",
        json.dumps(after_final, ensure_ascii=False, indent=2),
        "",
    ]
    (output_dir / "same_case_evolution_demo.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
