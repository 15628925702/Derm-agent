# Workflow Evolution

`workflow_evolution/` contains offline tools for generating, reviewing, and applying workflow configuration proposals.

Default runtime behavior is unchanged unless an approved proposal is explicitly enabled.

## Typical flow

1. Collect comparison reports.
2. Generate a proposal.
3. Review the proposal.
4. Approve it if appropriate.
5. Enable it explicitly at runtime when needed.

## Files

- `generate_proposal.py`: create proposal files from report inputs
- `proposal_generator.py`: proposal construction helpers
- `apply_proposal.py`: review/apply approved proposals
- `runtime.py`: optional runtime loading of approved proposals
- `doctor_experience_template.md`: optional reviewer input template
