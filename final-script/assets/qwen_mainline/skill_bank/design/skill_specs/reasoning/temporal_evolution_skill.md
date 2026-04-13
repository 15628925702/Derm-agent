# temporal_evolution_skill

## Purpose
Translate metadata history into structured temporal evidence so downstream reasoning can account for lesion behavior over time rather than treating the image as a timeless snapshot.

## When to Use
Use when metadata mentions growth, change, bleeding, recurrence, itch, pain, elevation change, or stability concerns.
Use whenever time-course may alter how current visual findings are interpreted.
Do not use this skill to convert history alone into a diagnosis.

## Clinical Pattern
This skill reconstructs:
- whether the process seems acute or chronic,
- whether change appears stable, slow, or rapid,
- whether recurrence is suggested,
- whether the overall trajectory is stable or unstable.

The emphasis is temporal framing, not disease naming.

## Strategy Overview
A dermatologist does not read morphology in isolation; time-course can change the meaning of the same visual pattern.
This skill should convert scattered history cues into a concise temporal frame while preserving uncertainty when metadata is sparse or noisy.
Its explainability value is that later reasoning can trace why "change over time" was considered important and exactly which history elements supported that view.

## Workflow
1. First gather all history-bearing metadata related to growth, change, bleeding, itch, pain, recurrence, or elevation.
2. Then separate strong temporal evidence from weak or nonspecific symptom language.
3. Then estimate whether the presentation feels acute, chronic, or indeterminate.
4. Then judge whether the reported course suggests stability, slow change, or rapid change.
5. Then note negative evidence, such as no documented progression or no reliable recurrence signal.
6. Finally preserve uncertainty if the metadata is sparse, contradictory, or too vague to support a strong time-course claim.

## Evidence to Check
- metadata fields for growth and change,
- bleeding or recurrent irritation history,
- itch and pain as secondary context,
- recurrence indicators,
- stability cues,
- negative evidence showing what temporal change is not actually documented.

## Watch Out For
- Absence of recorded change is not proof of true stability.
- Symptoms such as itch can be nonspecific and should not dominate interpretation.
- Do not let vague history language masquerade as rapid progression.
- Do not turn unstable evolution into an automatic disease label.

## Output Contract
Return structured temporal evidence only.

- `onset_type`: `acute`, `chronic`, or `unknown`
- `progression_speed`: `stable`, `slow`, `rapid`, or `unknown`
- `recurrence`: `yes`, `no`, or `uncertain`
- `stability`: `stable`, `unstable`, or `uncertain`

The output should improve CoT interpretability by making time-course reasoning explicit, bounded, and separate from final diagnosis selection.
