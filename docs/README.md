# DermAgent Docs

This directory is the stable home for project notes and experiment guides.

## Active docs

- `architecture/`: system design, self-evolution notes, and long-form architecture summaries.
- `experiments/`: experiment guides, migration notes, current result summaries, and 6x6 workflow status.
- `operations/`: runbooks and current-best usage notes.

## Archived notes

- `archive/legacy_iterations/`: older v3/v4/v7 development notes.
- `archive/workflow_notes/`: historical workflow refactor notes and stage-2 run notes.

## Runtime entry points

The active runtime folders stay at the repository root because many scripts and docs reference them directly:

- `scripts/`: low-level experiment, server, bootstrap, and analysis commands.
- `final-script/`: final-round wrapper entry points and model-specific env files.
- `final-score/`: final-round score/export area.
- `outputs/`: generated experiment outputs, ignored by git.
