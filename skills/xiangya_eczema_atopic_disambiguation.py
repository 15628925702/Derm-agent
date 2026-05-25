from __future__ import annotations

from typing import Any

from agent.label_space import canonicalize_label
from agent.state import CaseState
from skills.base import BaseSkill
from skills.catalog import make_skill_object
from skills.schema import SkillSchemaField, SkillStep, SkillTrigger


class XiangyaEczemaAtopicDisambiguationSkill(BaseSkill):
    name = "xiangya_eczema_atopic_disambiguation_skill"
    description = "Disambiguate Xiangya ECZEMA_DERMATITIS from ATOPIC_DERMATITIS using atopic-specific negative evidence."
    output_fields = (
        "candidate_focus",
        "generic_eczematous_morphology_presence",
        "atopic_specific_pattern_presence",
        "chronic_recurrent_proxy",
        "flexural_or_symmetric_pattern",
        "xerosis_or_lichenification",
        "evidence_supporting_eczema_dermatitis",
        "evidence_supporting_atopic_dermatitis",
        "evidence_against_direct_atopic_dermatitis",
        "retrieval_confusion_support",
        "memory_supporting_patterns",
        "canonical_label_recommendation",
        "recommendation_rationale",
        "confidence_under_current_evidence",
    )
    list_fields = (
        "candidate_focus",
        "evidence_supporting_eczema_dermatitis",
        "evidence_supporting_atopic_dermatitis",
        "evidence_against_direct_atopic_dermatitis",
        "memory_supporting_patterns",
    )
    skill_object = make_skill_object(
        skill_id="skill.xiangya_eczema_atopic_disambiguation.v1",
        name=name,
        description=description,
        skill_type="specialist",
        triggers=[
            SkillTrigger(
                condition=(
                    "Trigger for Xiangya 7-class rash cases where ATOPIC_DERMATITIS and "
                    "ECZEMA_DERMATITIS are both plausible or retrieved memory mentions this confusion."
                ),
                rationale=(
                    "Hulu-Med tends to promote generic eczematous rashes to atopic dermatitis; this skill must "
                    "separate nonspecific eczema morphology from truly atopic-specific evidence."
                ),
            )
        ],
        workflow_text=(
            "Focus narrowly on the Xiangya `ECZEMA_DERMATITIS` versus `ATOPIC_DERMATITIS` distinction. "
            "Treat `ATOPIC_DERMATITIS` as a narrower label that needs atopic-specific support such as flexural/symmetric pattern, "
            "clear xerosis/lichenification, chronic/recurrent history, pruritus, or atopy history. "
            "When the image only supports nonspecific erythematous/scaly/rough eczematous morphology and those atopic anchors are absent, "
            "prefer `ECZEMA_DERMATITIS` for the canonical recommendation. Do not use open-set labels."
        ),
        steps=[
            SkillStep("scope_pair", "Scope Pair", "Restrict reasoning to ECZEMA_DERMATITIS vs ATOPIC_DERMATITIS.", ["active Xiangya labels"]),
            SkillStep("detect_generic_eczema", "Detect Generic Eczema", "Identify erythema, scaling, crusting, rough surface, poorly defined patches, or nonspecific dermatitis.", ["eczematous morphology"]),
            SkillStep("check_atopic_anchors", "Check Atopic Anchors", "Look for flexural/symmetric pattern, xerosis, lichenification, chronic/recurrent proxy, itch, or atopy history.", ["atopic-specific support"]),
            SkillStep("use_memory", "Use Memory", "Check retrieved confusion memory for AD-to-eczema over-promotion caution.", ["retrieved confusion memory"]),
            SkillStep("recommend_canonical", "Recommend Canonical", "Recommend one Xiangya canonical label ID or uncertain.", ["canonical label ID"]),
        ],
        watch_outs=[
            "Do not upgrade generic eczema to ATOPIC_DERMATITIS without atopic-specific evidence.",
            "Do not use chronicity, recurrence, pruritus, or atopy history unless it is visible or present in metadata.",
            "Do not call psoriasis from scale alone.",
            "Use `ECZEMA_DERMATITIS` and `ATOPIC_DERMATITIS` exactly when recommending a canonical label.",
            "This skill produces structured evidence for fusion; it is not the final diagnosis step.",
        ],
        output_schema=[
            SkillSchemaField("candidate_focus", "list[str]", "Active Xiangya labels being compared."),
            SkillSchemaField("generic_eczematous_morphology_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("atopic_specific_pattern_presence", "str", "present, absent, or uncertain."),
            SkillSchemaField("chronic_recurrent_proxy", "str", "present, absent, or uncertain."),
            SkillSchemaField("flexural_or_symmetric_pattern", "str", "present, absent, or uncertain."),
            SkillSchemaField("xerosis_or_lichenification", "str", "present, absent, or uncertain."),
            SkillSchemaField("evidence_supporting_eczema_dermatitis", "list[str]", "Findings supporting nonspecific ECZEMA_DERMATITIS."),
            SkillSchemaField("evidence_supporting_atopic_dermatitis", "list[str]", "Findings supporting narrower ATOPIC_DERMATITIS."),
            SkillSchemaField("evidence_against_direct_atopic_dermatitis", "list[str]", "Missing or negative clues that argue against direct AD promotion."),
            SkillSchemaField("retrieval_confusion_support", "str", "present, absent, or uncertain memory support for this confusion."),
            SkillSchemaField("memory_supporting_patterns", "list[str]", "Retrieved memory patterns explicitly used."),
            SkillSchemaField("canonical_label_recommendation", "str", "One of ECZEMA_DERMATITIS, ATOPIC_DERMATITIS, or uncertain."),
            SkillSchemaField("recommendation_rationale", "str", "Short rationale for the canonical recommendation."),
            SkillSchemaField("confidence_under_current_evidence", "str", "low, medium, or high."),
        ],
    )

    def build_prompt(self, state: CaseState) -> str:
        return (
            f"{self.workflow_text()}\n"
            "You are executing the Xiangya eczema-vs-atopic disambiguation routine.\n"
            "When to use: Xiangya 7-class rash cases where generic eczema may be over-called as atopic dermatitis.\n"
            "What evidence to inspect: erythema, scale, crust, rough/poorly defined dermatitis, flexural or symmetric distribution, xerosis, lichenification, chronicity, pruritus, atopy history, and retrieved AD/Eczema confusion memory.\n"
            "Important: use Xiangya canonical labels only. If atopic-specific anchors are not actually available, do not recommend ATOPIC_DERMATITIS just because the rash is eczematous.\n"
            "Return compact JSON only with the following fields:\n"
            f"{self.output_schema_text()}\n"
            f"Initial perception: {self.perception_snapshot(state)}\n"
            f"Baseline diagnosis: {self._diagnosis_snapshot(state.baseline_diagnosis)}\n"
            f"Current skill outputs: {self.skill_outputs_snapshot(state, preferred_skills=('morphology_analysis_skill', 'color_pattern_analysis_skill', 'border_surface_analysis_skill', 'distribution_analysis_skill', 'differential_compare_skill', 'xiangya_acne_disambiguation_skill'), max_skills=6)}\n"
            f"Metadata: {self.metadata_snapshot(state)}\n"
        )

    def normalize_output(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        normalized = super().normalize_output(raw_output)
        recommendation = str(normalized.get("canonical_label_recommendation", "")).strip()
        canonical = canonicalize_label(
            recommendation,
            label_space_id="xiangya_7class",
            dataset_name="xiangya_7class",
        )
        if canonical in {"ECZEMA_DERMATITIS", "ATOPIC_DERMATITIS"}:
            normalized["canonical_label_recommendation"] = canonical
        else:
            normalized["canonical_label_recommendation"] = "uncertain"
        if normalized.get("recommendation_type") == "descriptive_evidence":
            normalized["recommendation_type"] = "comparative_support"
        return normalized

    def after_execute(self, state: CaseState, output: dict[str, Any]) -> None:
        derived = self._deterministic_eczema_readout(state)
        if not derived:
            return
        output.update(derived)
        state.skill_outputs[self.name] = output

    def normalize_field(self, field_name: str, value: Any) -> Any:
        if field_name in {
            "generic_eczematous_morphology_presence",
            "atopic_specific_pattern_presence",
            "chronic_recurrent_proxy",
            "flexural_or_symmetric_pattern",
            "xerosis_or_lichenification",
            "retrieval_confusion_support",
        }:
            return self._normalize_presence(value)
        if field_name == "confidence_under_current_evidence":
            normalized = str(value).strip().lower()
            if normalized in {"low", "medium", "high"}:
                return normalized
            return "medium" if value else "unknown"
        return super().normalize_field(field_name, value)

    @staticmethod
    def _diagnosis_snapshot(output: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(output or {})
        return {
            "final_diagnosis": payload.get("final_diagnosis"),
            "differential_diagnoses": payload.get("differential_diagnoses", []),
            "rationale": payload.get("rationale"),
            "confidence": payload.get("confidence"),
        }

    @staticmethod
    def _normalize_presence(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"present", "yes", "true", "seen", "visible", "positive"}:
            return "present"
        if text in {"absent", "no", "false", "not seen", "none", "negative"}:
            return "absent"
        return "uncertain"

    @staticmethod
    def _deterministic_eczema_readout(state: CaseState) -> dict[str, Any]:
        workflow_context = state.case_input.workflow_context or {}
        if str(workflow_context.get("workflow_cell_id", "")).strip().lower() != "hulumed__xiangya_7class__retrieval_open_v1":
            return {}
        baseline = state.baseline_diagnosis or {}
        baseline_label = canonicalize_label(
            baseline.get("final_diagnosis"),
            label_space_id="xiangya_7class",
            dataset_name="xiangya_7class",
        )
        baseline_text = XiangyaEczemaAtopicDisambiguationSkill._safe_join(
            [
                baseline.get("final_diagnosis"),
                baseline.get("rationale"),
                baseline.get("differential_diagnoses"),
            ]
        )
        skill_text = XiangyaEczemaAtopicDisambiguationSkill._safe_join(
            [
                state.perception,
                {
                    key: value
                    for key, value in (state.skill_outputs or {}).items()
                    if key
                    in {
                        "morphology_analysis_skill",
                        "color_pattern_analysis_skill",
                        "border_surface_analysis_skill",
                        "distribution_analysis_skill",
                        "differential_compare_skill",
                        "metadata_consistency_skill",
                    }
                },
            ]
        )
        metadata = state.clinical_metadata or {}
        metadata_text = XiangyaEczemaAtopicDisambiguationSkill._safe_join(
            [
                metadata.get("related_category"),
                metadata.get("presentation_mode_hint"),
                metadata.get("body_site_hint"),
                metadata.get("itch"),
                metadata.get("grew"),
                metadata.get("changed"),
            ]
        )
        retrieval_patterns = XiangyaEczemaAtopicDisambiguationSkill._retrieval_patterns(state)
        retrieval_support = bool(retrieval_patterns)
        combined = f"{baseline_text} {skill_text} {metadata_text}".lower()
        skill_lower = skill_text.lower()
        metadata_lower = metadata_text.lower()
        distribution_output = state.skill_outputs.get("distribution_analysis_skill", {})
        distribution_lower = XiangyaEczemaAtopicDisambiguationSkill._safe_join(distribution_output).lower()

        generic_terms = (
            "eczema",
            "eczematous",
            "dermatitis",
            "erythema",
            "erythematous",
            "scaling",
            "scale",
            "crust",
            "rough",
            "poorly defined",
            "patch",
            "rash",
        )
        eczema_hits = sum(1 for term in generic_terms if term in combined)
        flexural = "flexural" in combined
        lichenification = "lichenif" in combined
        metadata_chronic = any(term in metadata_lower for term in ("chronic", "recurrent", "itch", "pruritus", "atopy"))
        visible_xerosis = "xerosis" in skill_lower or "dry" in skill_lower
        symmetric = "symmetric" in skill_lower or "symmetry" in skill_lower
        atopic_strong = flexural or lichenification or metadata_chronic or (visible_xerosis and symmetric)
        focal_or_asymmetric = any(
            term in distribution_lower
            for term in ("asymmetric", "localized", "scattered")
        )
        baseline_in_pair = baseline_label in {"ATOPIC_DERMATITIS", "ECZEMA_DERMATITIS"} or any(
            term in baseline_text.lower() for term in ("atopic dermatitis", "eczema", "eczematous dermatitis")
        )
        if not baseline_in_pair or eczema_hits < 1:
            return {}

        eczema_support = []
        if any(term in combined for term in ("erythema", "erythematous")):
            eczema_support.append("erythematous dermatitis morphology")
        if any(term in combined for term in ("scaling", "scale", "rough", "crust")):
            eczema_support.append("rough/scaly eczematous surface")
        if any(term in combined for term in ("poorly defined", "patch", "rash")):
            eczema_support.append("nonspecific patch/rash pattern")
        if retrieval_support:
            eczema_support.append("retrieved AD-vs-eczema confusion memory cautions against direct AD promotion")

        atopic_support = []
        if flexural:
            atopic_support.append("flexural pattern mentioned")
        if lichenification:
            atopic_support.append("lichenification mentioned")
        if visible_xerosis:
            atopic_support.append("xerosis/dryness mentioned")
        if symmetric:
            atopic_support.append("symmetric pattern mentioned")
        if metadata_chronic:
            atopic_support.append("history or symptom proxy supports atopic dermatitis")

        if atopic_strong:
            recommendation = "ATOPIC_DERMATITIS"
            against_ad = ["atopic-specific anchors are present, so direct AD promotion is not contradicted"]
            confidence = "medium"
            rationale = "Atopic-specific anchors are present, so the AD label should be preserved."
        else:
            recommendation = "ECZEMA_DERMATITIS" if retrieval_support and focal_or_asymmetric else "uncertain"
            against_ad = [
                "no flexural or lichenified atopic pattern available",
                "no metadata-backed chronic/recurrent/pruritic atopy proxy available",
            ]
            confidence = "high" if retrieval_support and eczema_hits >= 2 else "medium"
            rationale = (
                "Generic eczematous morphology is present without atopic-specific anchors, with focal/asymmetric distribution and retrieved memory support for preserving the AD-vs-eczema distinction."
                if retrieval_support and focal_or_asymmetric
                else "Generic eczematous morphology is present, but distribution/memory support is not strong enough for a canonical rescue."
            )

        return {
            "candidate_focus": ["ECZEMA_DERMATITIS", "ATOPIC_DERMATITIS"],
            "generic_eczematous_morphology_presence": "present" if eczema_hits else "uncertain",
            "atopic_specific_pattern_presence": "present" if atopic_strong else "absent",
            "chronic_recurrent_proxy": "present" if metadata_chronic else "absent",
            "flexural_or_symmetric_pattern": "present" if flexural or symmetric else "absent",
            "xerosis_or_lichenification": "present" if visible_xerosis or lichenification else "absent",
            "evidence_supporting_eczema_dermatitis": eczema_support[:5] or ["nonspecific eczematous dermatitis morphology"],
            "evidence_supporting_atopic_dermatitis": atopic_support[:5],
            "evidence_against_direct_atopic_dermatitis": against_ad[:5],
            "retrieval_confusion_support": "present" if retrieval_support else "absent",
            "memory_supporting_patterns": retrieval_patterns[:4],
            "canonical_label_recommendation": recommendation,
            "recommendation_rationale": rationale,
            "confidence_under_current_evidence": confidence,
        }

    @staticmethod
    def _retrieval_patterns(state: CaseState) -> list[str]:
        patterns: list[str] = []
        bundle = state.retrieval_bundle or {}
        for key in ("abstract_results", "tactical_results", "skill_summary", "aggregator_summary"):
            for record in bundle.get(key, []) or []:
                text = XiangyaEczemaAtopicDisambiguationSkill._safe_join(
                    [
                        record.get("confusion_pair"),
                        record.get("experience_type"),
                        record.get("perception_summary"),
                        record.get("learning_points"),
                        record.get("pattern_summary"),
                    ]
                ).lower()
                if "atopic" in text and "eczema" in text:
                    source_id = str(record.get("source_id") or record.get("id") or "retrieved_memory").strip()
                    if source_id and source_id not in patterns:
                        patterns.append(source_id)
        return patterns

    @staticmethod
    def _safe_join(values: Any) -> str:
        chunks: list[str] = []
        if isinstance(values, dict):
            values = values.values()
        if isinstance(values, (list, tuple, set)):
            for value in values:
                chunks.append(XiangyaEczemaAtopicDisambiguationSkill._safe_join(value))
            return " ".join(item for item in chunks if item)
        if values is None:
            return ""
        return str(values)
