# escalation_recommendation_skill

## Purpose
Decide whether the current evidence state warrants more cautious or escalated diagnostic checking.

## When to Use
Use when risk, unresolved uncertainty, or contradictions remain clinically meaningful.
Use when the system should explicitly ask whether further checking is needed before confident closure.
Do not use this skill to recommend treatment.

## Clinical Pattern
This skill focuses on:
- whether escalation is needed,
- why it is needed,
- what diagnostic check type would best reduce unresolved risk or uncertainty,
- what caution flags should travel forward.

Its value is that it formalizes a common physician move: deciding when the current evidence state is not enough for comfortable closure.

## Strategy Overview
Doctors often decide not only what they think, but whether the current information state requires a more cautious next step.
This skill should therefore integrate risk, uncertainty, conflicts, and missing information into an escalation judgment without giving therapy advice.
Its explainability value is strong because final reasoning and reflection can point back to explicit escalation logic.

## Workflow
1. First review malignancy concern, uncertainty, contradictions, and information gaps together.
2. Then decide whether escalation is needed, should be considered, or is not currently needed.
3. Then name the most appropriate next diagnostic check type rather than a treatment action.
4. Then record caution flags that justify continued careful interpretation.
5. Finally preserve uncertainty and avoid overstating the need for escalation when evidence is still limited.

## Evidence to Check
- malignancy risk outputs,
- uncertainty outputs,
- contradiction outputs,
- information-gap outputs,
- exclusion and differential outputs,
- clinically relevant metadata context.

## Watch Out For
- Do not give treatment advice.
- Do not escalate for vague reasons without tying it to risk, conflict, or missing information.
- Do not suppress caution when nontrivial risk and missing evidence coexist.
- Do not convert escalation reasoning into final diagnosis.

## Output Contract
Return structured escalation reasoning only.

- `whether_escalation_needed`: `yes`, `consider`, or `no`
- `escalation_reason`: why escalation is needed or should be considered
- `suggested_next_check_type`: the type of next diagnostic check
- `caution_flags`: specific reasons to remain cautious

The output should improve CoT transparency by making escalation logic explicit and auditable.
