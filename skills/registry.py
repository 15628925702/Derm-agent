from __future__ import annotations

from agent.state import CaseState
from integrations.openai_client import DermOpenAIClient
from skills.base import BaseSkill
from skills.schema import SkillObject
from skills.ack_scc_specialist import AckSccSpecialistSkill
from skills.benign_mimic_specialist import BenignMimicSpecialistSkill
from skills.border_surface import BorderSurfaceAnalysisSkill
from skills.color_pattern import ColorPatternAnalysisSkill
from skills.contradiction_check import ContradictionCheckSkill
from skills.differential_compare import DifferentialCompareSkill
from skills.distribution import DistributionAnalysisSkill
from skills.escalation_recommendation import EscalationRecommendationSkill
from skills.exclusion_reasoning import ExclusionReasoningSkill
from skills.information_gap_detection import InformationGapDetectionSkill
from skills.malignancy_risk import MalignancyRiskAssessmentSkill
from skills.lesion_description_structuring import LesionDescriptionStructuringSkill
from skills.metadata_consistency import MetadataConsistencySkill
from skills.mel_nev_specialist import MelNevSpecialistSkill
from skills.morphology import MorphologyAnalysisSkill
from skills.temporal_evolution import TemporalEvolutionSkill
from skills.uncertainty import UncertaintyAssessmentSkill
from skills.xiangya_acne_disambiguation import XiangyaAcneDisambiguationSkill
from skills.xiangya_eczema_atopic_disambiguation import XiangyaEczemaAtopicDisambiguationSkill


class SkillRegistry:
    def __init__(self, skills: list[BaseSkill]) -> None:
        self._skills = {skill.name: skill for skill in skills}
        self._skills_by_id = {skill.get_skill_object().skill_id: skill for skill in skills}

    def get(self, skill_name: str) -> BaseSkill:
        return self._skills[skill_name]

    def get_by_skill_id(self, skill_id: str) -> BaseSkill:
        return self._skills_by_id[skill_id]

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def list_skill_objects(self) -> list[dict]:
        return [skill.to_skill_dict() for skill in self._skills.values()]

    def list_skill_objects_structured(self) -> list[SkillObject]:
        return [skill.get_skill_object() for skill in self._skills.values()]

    def list_skill_objects_by_names(self, skill_names: list[str]) -> list[SkillObject]:
        result: list[SkillObject] = []
        seen: set[str] = set()
        for skill_name in skill_names:
            normalized = str(skill_name).strip()
            if not normalized or normalized in seen or normalized not in self._skills:
                continue
            seen.add(normalized)
            result.append(self._skills[normalized].get_skill_object())
        return result

    def list_skill_objects_by_ids(self, skill_ids: list[str]) -> list[SkillObject]:
        result: list[SkillObject] = []
        seen: set[str] = set()
        for skill_id in skill_ids:
            normalized = str(skill_id).strip()
            if not normalized or normalized in seen or normalized not in self._skills_by_id:
                continue
            seen.add(normalized)
            result.append(self._skills_by_id[normalized].get_skill_object())
        return result

    def run_skill(self, skill_name: str, state: CaseState, client: DermOpenAIClient) -> dict:
        skill = self.get(skill_name)
        return skill.execute(state, client)

    def run_many(self, skill_names: list[str], state: CaseState, client: DermOpenAIClient) -> dict[str, dict]:
        return {skill_name: self.run_skill(skill_name, state, client) for skill_name in skill_names}


def build_default_registry() -> SkillRegistry:
    skills: list[BaseSkill] = [
        MorphologyAnalysisSkill(),
        ColorPatternAnalysisSkill(),
        BorderSurfaceAnalysisSkill(),
        DistributionAnalysisSkill(),
        LesionDescriptionStructuringSkill(),
        TemporalEvolutionSkill(),
        MetadataConsistencySkill(),
        DifferentialCompareSkill(),
        ExclusionReasoningSkill(),
        InformationGapDetectionSkill(),
        ContradictionCheckSkill(),
        MelNevSpecialistSkill(),
        AckSccSpecialistSkill(),
        BenignMimicSpecialistSkill(),
        XiangyaAcneDisambiguationSkill(),
        XiangyaEczemaAtopicDisambiguationSkill(),
        MalignancyRiskAssessmentSkill(),
        UncertaintyAssessmentSkill(),
        EscalationRecommendationSkill(),
    ]
    return SkillRegistry(skills)
