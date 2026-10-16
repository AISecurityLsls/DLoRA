# Synthetic KD inputs

Pre-generated Self-Instruct inputs are not committed to this repository.
Generate or refresh them for one dataset with:

```bash
bash scripts/synthetic/generate_self_instruct_inputs.sh <dataset>
```

The output is written to
`synthetic_kd_inputs/self_instruct/<dataset>/latest.inputs.jsonl`.
