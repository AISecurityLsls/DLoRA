# DormantLoRA

Code repository for the paper that introduces DormantLoRA (submitted). This is
defense-oriented work, and the paper studies the corresponding defense method.

## Installation

```bash
cd DormantLoRA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA version if the default wheel
is not suitable.

## Base models

By default the code looks for models under:

```text
models/Llama-3-8B/
models/Qwen2.5-7B/
models/Mistral-7B/
```

If your models live elsewhere, copy the example path config and edit it:

```bash
cp configs/paths.example.json configs/paths.local.json
```

`configs/paths.local.json` is gitignored. All `dataset_paths`,
`base_model_paths`, `target_adapter_root`, `dormant_adapter_output_root`, and
`log_root` values can be overridden there.

## Data

The benchmark datasets are not committed to this repository. Download the
CL_Benchmark Long-Sequence tasks and place them under `data/Long_Sequence/` as
described in [data/README.md](data/README.md).

## Quick start

### 1. Train a clean Target LoRA

```bash
CUDA_ID=0 BASE_MODEL_NAME=Llama-3-8B \
bash scripts/run_train_target_lora.sh agnews
```

This writes the adapter to:

```text
adapters/TargetModelAdapters/Llama-3-8B/lora_agnews/
```

Train the four Target datasets in a bank:

```bash
CUDA_ID=0 bash scripts/train_target_loras.sh
```

### 2. Generate KD inputs

Pre-generated KD inputs are not committed. Generate them with an
OpenAI-compatible endpoint in `.env`:

```bash
cp .env.example .env
# edit LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
DRY_RUN=1 bash scripts/synthetic/generate_self_instruct_inputs.sh agnews
```

Outputs are written under
`synthetic_kd_inputs/self_instruct/<target_dataset>/`. Remove `DRY_RUN=1`
after checking the prompt to perform the real generation.

### 3. Train a Dormant LoRA

```bash
CUDA_ID=0 PATH_CONFIG=configs/paths.local.json \
bash scripts/run_train_dormant_lora.sh mnli agnews
```

Here:

- `mnli` is the Dormant dataset used to train and evaluate the Dormant LoRA.
- `agnews` selects the frozen Target LoRA and its KD inputs.

The default run optimizes the full Dormant LoRA with the three DormantLoRA
components:

1. Anti-Backdoor Training,
2. Backdoor Training,
3. Target Adapter Capability Preservation.

No rank-partition mode needs to be selected; the optimizer always updates the
whole Dormant LoRA.

### KD source

```bash
KD_SOURCE=synthetic_self_instruct CUDA_ID=0 \
bash scripts/run_train_dormant_lora.sh mnli agnews
```

Supported values are `synthetic_self_instruct`, `target_train`, `target_valid`,
and `none`. `none` disables KD.

## Three-part ablation

Run the three-component ablation for one pair:

```bash
CUDA_ID=0 bash scripts/ablation.sh mnli agnews
```

This runs:

| Mode | Removed component |
| --- | --- |
| `no_anti_backdoor` | Anti-Backdoor Training |
| `no_backdoor` | Backdoor Training |
| `no_preservation` | Target Adapter Capability Preservation |

Run one mode only with:

```bash
CUDA_ID=0 bash scripts/ablation.sh mnli agnews no_preservation
```

Each mode writes its Dormant adapter under
`adapters/DormantModelAdapters/<base>/ablation_<mode>/`.

## Outputs

Dormant training writes logs under:

```text
logs/<base>/[<param_group_path>/]<dormant>-with-<target>/
```

and saves the Dormant adapter under:

```text
adapters/DormantModelAdapters/<base>/[<param_group_path>/]lora_<dormant>-with-<target>/
```

For each run the main metrics are also stored in `run_meta.json` inside the
adapter directory.
