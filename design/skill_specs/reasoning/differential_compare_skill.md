# differential_compare_skill

## Purpose
Compare leading diagnostic candidates in a structured way without allowing the agent to choose the final diagnosis.

## When to Use
Use when initial perception has produced two or more plausible differential candidates.
Use after core observation skills have created enough structured evidence to support comparison.
Do not use this skill to select a winner or introduce unsupported new disease names.

## Clinical Pattern
This skill is about pairwise comparison:
- what features support candidate A over candidate B,
- what features support the opposite side,
- what unresolved evidence prevents clean separation,
- what negative evidence weakens a candidate even if it remains in scope.

The function is narrowing and clarifying, not deciding.

## Strategy Overview
In clinical reasoning, differential comparison is the stage where physicians test competing explanations against observed evidence.
This skill should keep the comparison open while making support and counter-support visible.
Its explainability value is high because it exposes why multiple hypotheses remain plausible and which clues are carrying the comparison instead of burying that process inside an opaque conclusion.

## Workflow
1. First select the most relevant candidate pairs from the current differential, usually focusing on the closest or most consequential comparison.
2. Then gather evidence from observation, temporal, and risk-oriented skills that supports one side of the pair.
3. Then actively gather negative evidence or counter-evidence that weakens each side.
4. Then state what remains unresolved and why the comparison cannot yet be cleanly closed.
5. Then avoid premature winner selection even if one side currently looks stronger.
6. Finally preserve diagnostic openness so Qwen remains the only final decision maker.

## Evidence to Check
- current differential candidates from initial perception,
- morphology, color, border, distribution, and temporal outputs,
- metadata consistency findings,
- explicit negative evidence against each candidate,
- unresolved or contradictory clues that keep both candidates alive.

## Watch Out For
- Do not collapse comparison into final diagnosis selection.
- Do not add unsupported disease names that were not already in scope.
- Do not hide counter-evidence just because one candidate looks stronger.
- Do not present a ranking as if it were a final verdict.

## Output Contract
Return structured comparative evidence only.

- `candidate_pairs`: list of pairwise comparisons such as `A vs B`
- `supporting_evidence`: clues that support one side of the comparison
- `conflicting_evidence`: clues that oppose, weaken, or keep the comparison unresolved

The output should strengthen CoT interpretability by making pairwise clinical reasoning explicit while preserving final diagnostic openness.
