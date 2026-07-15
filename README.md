# DermAgent

DermAgent is a structured reasoning and evaluation framework for dermatology image tasks.

The repository combines:

- case loading and dataset adapters
- workflow/profile selection
- skill execution and evidence organization
- evaluation runners
- export utilities for analysis artifacts

## Repository layout

- `agent/`: core reasoning, routing, evaluation, and export logic
- `dataio/`: dataset loaders and schema alignment helpers
- `skills/`: reusable reasoning skills
- `memory/`: experience and transformation utilities
- `cognition/`: cross-case state utilities
- `scripts/`: experiment and maintenance scripts
- `workflow_evolution/`: offline proposal tooling for workflow configuration changes
- `tests/`: automated tests

## Evaluation

The evaluation scripts support fixed split inputs, reproducible execution, and isolated output directories.

## Notes

- Configuration and routing behavior are implemented in code and config files under `agent/` and `configs/`.
- Auxiliary documents in the repository are operational references and can be updated independently from runtime behavior.
