# contradiction_check_skill

## Purpose
Audit the current reasoning state for internal inconsistency, missing support, and unresolved gaps before evidence is handed downstream.

## When to Use
Use after multiple evidence streams are active or when reasoning still feels unstable despite several completed skills.
Use when image, metadata, and skill outputs may be pulling in different directions.
Do not use this skill to relabel the lesion; use it to expose reasoning problems.

## Clinical Pattern
This skill looks for three kinds of problems:
- direct contradictions between evidence sources,
- missing links where a conclusion lacks support,
- broader reasoning gaps where clinically relevant information is still absent.

The target is reasoning integrity, not diagnosis.

## Strategy Overview
Dermatologist-style reasoning is iterative constraint reduction, which means contradictions and unsupported jumps must be visible rather than silently ignored.
This skill should behave like a consistency auditor that tests whether the current story actually hangs together.
Its explainability value is central: it makes hidden breaks in the reasoning chain explicit so later integration can stay cautious and clinically interpretable.

## Workflow
1. First compare perception, metadata, and prior skill outputs for direct factual conflict.
2. Then trace whether any claim in the current reasoning state depends on missing intermediate support.
3. Then distinguish contradiction from ordinary uncertainty so weak evidence is not over-penalized.
4. Then identify broader reasoning gaps, such as missing lesion history, insufficient scale, or unresolved image ambiguity.
5. Then record negative evidence, such as no clear conflict between two evidence streams where one might have been expected.
6. Finally summarize what still blocks a clean reasoning chain without drifting into final diagnosis.

## Evidence to Check
- initial perception content,
- metadata fields,
- outputs from earlier skills,
- agreement or conflict between observation and reasoning layers,
- unsupported inference jumps,
- negative evidence showing where apparent conflict is not actually established.

## Watch Out For
- Do not invent contradictions that are really just uncertainty.
- A reasoning gap is not always a factual conflict.
- Do not use this skill to argue for a disease label.
- Do not overclaim missing links when the needed evidence was never available.

## Output Contract
Return structured reasoning-audit evidence only.

- `contradictions`: direct conflicts between evidence sources
- `missing_links`: missing support that prevents a clean reasoning chain
- `reasoning_gaps`: broader unresolved weaknesses in the current case reasoning state

The output should improve CoT interpretability and safety by exposing where the evidence chain is broken, weak, or underdetermined before final Qwen integration.
