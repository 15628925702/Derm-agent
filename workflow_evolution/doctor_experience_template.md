# Doctor Experience Intake Template

This file is optional input for workflow evolution proposal generation. It is
plain text on purpose: a physician can write clinical rules without editing
code. Generated proposals remain disabled until reviewed and approved.

## Context

- Model:
- Dataset / hospital source:
- Label space:
- Date:
- Reviewer:

## Observed Failure Patterns

- Baseline often predicts `<predicted disease/family>` when the reference or senior review is `<target disease/family>`.
- Agent over-calls `<disease/family>` when `<visual clue>` is present.
- Agent under-calls `<disease/family>` when `<visual clue>` is present.

## Clinical Rules From Physician Experience

- If `<visual morphology/location/history>` is present, consider `<disease/family>` more strongly than `<common wrong disease/family>`.
- Do not override toward `<disease/family>` when `<danger signal or contraindicating feature>` is present.
- For malignant-risk cases, keep baseline or require dermatology follow-up when `<risk feature>` appears.

## Skill Suggestions

- Add or strengthen a skill that checks `<specific morphology or location>`.
- Disable or downweight `<skill name>` for this workflow when it repeatedly causes `<failure mode>`.
- Add a specialist comparison for `<disease A> vs <disease B>`.

## Conservative Fusion Suggestions

- Loosen only this direction: `<baseline family> -> <agent family>`.
- Tighten this direction: `<baseline family> -> <agent family>`.
- Minimum evidence threshold should be:
  - selected evidence present:
  - uncertainty level:
  - support margin:
  - subtype support margin:
  - contradiction count:

## Review Decision

- Approve candidate workflow:
- Reject candidate workflow:
- Need more validation:
- Risk notes:
