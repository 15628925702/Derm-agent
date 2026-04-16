# Final Runs

This directory is the large-results store for final experiment outputs.

## Recommended management

- Keep this directory out of the main GitHub source repo history.
- Upload it to a private Hugging Face dataset repo when you want remote backup or cross-machine access.

## Upload helper

Example:

```bash
HF_REPO_ID="your-user/dermagent-final-runs" \
HF_TOKEN="hf_xxx" \
/root/DermAgent/scripts/upload_final_runs_to_hf.sh
```

Important environment variables:

- `HF_REPO_ID`: target Hugging Face dataset repo, for example `your-user/dermagent-final-runs`
- `HF_TOKEN`: optional if you already ran `hf auth login`
- `LOCAL_PATH`: defaults to this directory
- `REVISION`: defaults to `main`
- `PRIVATE=1`: create the repo as private if it does not already exist
