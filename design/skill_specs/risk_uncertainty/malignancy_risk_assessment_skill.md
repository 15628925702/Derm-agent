# malignancy_risk_assessment_skill

## Purpose
Estimate how concerning the current lesion appears from a risk perspective without converting that concern into a final malignant diagnosis.

## When to Use
Use when current evidence leaves malignancy plausibly in scope or when concerning visual or history cues are present.
Use after enough observation and temporal evidence exists to support risk framing.
Do not use this skill as a hidden classifier for cancer.

## Clinical Pattern
This skill asks:
- are there visible or historical alarm signals,
- how strong is the cumulative concern,
- what risk level best describes the current state of evidence,
- which negative findings reduce concern even if risk cannot be dismissed.

The target is risk framing, not disease naming.

## Strategy Overview
Clinicians often separate "how dangerous could this be" from "what exactly is it."
This skill should aggregate concerning and reassuring clues into a low, medium, or high risk summary while remaining explicit about why.
Its explainability value is that downstream reasoning can reference named alarm signals and balancing negative evidence rather than relying on an opaque overall impression.

## Workflow
1. First collect concerning visual cues from morphology, color, border, and surface findings.
2. Then add history-based alarm signals such as change, bleeding, or instability when available.
3. Then actively look for negative evidence that lowers concern, such as absence of convincing irregularity, absence of strong temporal change, or absence of alarming surface disruption.
4. Then weigh the whole pattern and assign a coarse risk level without naming a specific cancer.
5. Then state the alarm signals explicitly so later reasoning can inspect them.
6. Finally preserve uncertainty if risk is elevated mainly because key information is missing rather than because strong malignant evidence is present.

## Evidence to Check
- irregular border or asymmetric pigment,
- concerning surface change,
- unstable or changing history,
- bleeding or other alarm symptoms,
- outputs from earlier observation and temporal skills,
- negative evidence showing what high-risk features are not convincingly present.

## Watch Out For
- Risk level is not a final diagnosis.
- Do not suppress reassuring evidence when one alarming cue appears.
- Symptoms can raise caution but are often nonspecific.
- Do not let missing information automatically inflate risk without explaining why.

## Output Contract
Return structured risk evidence only.

- `risk_level`: `low`, `medium`, `high`, or `unknown`
- `risk_evidence`: list of structured reasons supporting the current risk framing
- `alarm_signals`: explicit concerning features worth carrying into downstream integration

The output should improve CoT interpretability by showing exactly why the case was framed as more or less concerning, while keeping final diagnosis outside the skill boundary.
