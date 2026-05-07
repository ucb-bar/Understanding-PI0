# Understanding PI0 / SmolVLA MX Export

This branch contains the SmolVLA MX quantization and IREE export workflow.

It is focused on:
- inspecting the SmolVLA module structure and quantization plan
- applying TorchAO MX FP8 as much as possible, with fallback for unsupported layers
- validating one-step numerics after quantization
- exporting a one-step SmolVLA wrapper to Torch-MLIR / IREE via `iree-turbine`

## Repository layout

- `scripts/quantizing_smolvla/inspect_fqns.py`  
  Print all `nn.Linear` FQNs, shapes, dtypes, bucket assignment, and quantization plan.

- `scripts/quantizing_smolvla/quantize_mx.py`  
  Apply the quantization recipe and save a checkpoint payload plus JSON report.

- `scripts/quantizing_smolvla/validate_one_step.py`  
  Compare one-step baseline vs quantized outputs.

- `scripts/export_iree.py`  
  Apply IREE/Turbine dtype patches, rewrite quantized linears into exportable wrappers, and export MLIR or VMFB.

- `scripts/export_executorch.py`  
  Lower the one-step wrapper to an ExecuTorch `.pte` with PT2E int8 quantization through the XNNPACK delegate (CPU runtime).

- `understanding_pi0/common/iree_ocp_patch.py`  
  Runtime patches for IREE Turbine / FX importer dtype transport and importer compatibility.

- `understanding_pi0/common/mx_exportable.py`  
  Export-time wrappers for quantized linears:
  - MX linears stay in MX storage and dequantize explicitly in forward
  - non-MX quantized linears are rewritten to dequantized export wrappers

## Requirements

- Python 3.12
- `uv`
- CUDA-capable PyTorch environment if exporting from GPU
- sibling checkout of `lerobot` at `../lerobot`

This repo uses an editable source dependency:

```toml
[tool.uv.sources]
lerobot = { path = "../lerobot", editable = true }
```

This repository contains some basic scripts to understand the [PI0 policy](https://github.com/Physical-Intelligence/openpi).


## Setting up the repository

We use the Huggingface implementation of PI0 policy. In order to look into the source code, it is recommended to pull the lerobot and transformers source code.

```bash
mkdir -p ../third_party
cd ../third_party

git clone -b mlir-smolvla https://github.com/ucb-bar/Understanding-PI0.git
git clone https://github.com/huggingface/lerobot.git
```

Then install:

```bash
cd Understanding-PI0
uv python pin 3.12
uv sync --extra export_iree
```

## E2E workflow

1. Inspect the quantization plan

```bash
uv run python scripts/quantizing_smolvla/inspect_fqns.py \
  --model-id lerobot/smolvla_base \
  --device cuda
```

2. Quantize and save artifacts

```bash
uv run python scripts/quantizing_smolvla/quantize_mx.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --out reports/smolvla_mx/smolvla_mx_quantized.pt \
  --report-json reports/smolvla_mx/quant_report.json
```

Optional smoke test:

```bash
uv run python scripts/quantizing_smolvla/quantize_mx.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --smoke-test
```

3. Validate one-step numerics

```bash
uv run python scripts/quantizing_smolvla/validate_one_step.py \
  --model-id lerobot/smolvla_base \
  --device cuda
```

4. Export MLIR

```bash
uv run python scripts/export_iree.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --print-readable \
  --out reports/smolvla_mx/smolvla_one_step.mlir
```

5. Optionally compile VMFB

```bash
uv run python scripts/export_iree.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --compile-vmfb \
  --out reports/smolvla_mx/smolvla_one_step.mlir \
  --vmfb-out reports/smolvla_mx/smolvla_one_step.vmfb
```

### Expected Outputs

After a successful run you should see:

- `reports/smolvla_mx/quant_report.json`
- `reports/smolvla_mx/smolvla_mx_quantized.pt`
- `reports/smolvla_mx/smolvla_one_step.mlir`

## ExecuTorch (.pte) export — int8

`scripts/export_executorch.py` lowers the same one-step wrapper used by the
IREE flow to an ExecuTorch `.pte` for the XNNPACK CPU delegate, with
post-training int8 quantization through the **PT2E** flow
(`torch.export` → `prepare_pt2e` → calibration → `convert_pt2e`).

> **Why a parallel venv?** ExecuTorch 1.1.x pins `torchao==0.15` while the
> IREE/MX flow in this repo pins `torchao==0.16`. The two cannot coexist
> in a single resolved environment, so we keep a dedicated
> `.venv-executorch/` next to the default `.venv/` and switch by pointing
> `python` at the right interpreter.

### 1. Create the parallel venv

From the repo root (and with a sibling `lerobot` checkout at `../lerobot`,
same as for the IREE flow):

```bash
# Create the parallel venv
uv venv .venv-executorch --python 3.12

# Reusable shorthand for the rest of this section
EVDEV_OVERRIDE=$(printf "evdev ; sys_platform == 'none'\n")

# 1) Install ExecuTorch + ml-dtypes + safetensors. The --override silences
#    the evdev / pynput build issue documented in pyproject.toml [tool.uv]
#    (same workaround the default `uv sync` uses).
VIRTUAL_ENV="$(pwd)/.venv-executorch" uv pip install \
  --python .venv-executorch/bin/python \
  --override <(echo "$EVDEV_OVERRIDE") \
  "executorch>=1.1.0,<2.0" \
  ml-dtypes safetensors

# 2) Install lerobot[smolvla] EDITABLE from the sibling checkout. A non-
#    editable file:// install copies the source into site-packages and
#    confuses the lerobot.policies package layout, so use -e here.
VIRTUAL_ENV="$(pwd)/.venv-executorch" uv pip install \
  --python .venv-executorch/bin/python \
  --override <(echo "$EVDEV_OVERRIDE") \
  -e "../lerobot[smolvla]"

# 3) Pin transformers==5.3.0. Newer transformers (5.8+) makes
#    PretrainedConfig a dataclass with default fields, which trips the
#    `non-default argument follows default argument` error in
#    lerobot.policies.groot.groot_n1.GR00TN15Config and crashes the
#    `from lerobot.policies.smolvla...` import we use here.
VIRTUAL_ENV="$(pwd)/.venv-executorch" uv pip install \
  --python .venv-executorch/bin/python \
  --override <(echo "$EVDEV_OVERRIDE") \
  "transformers==5.3.0"
```

This pulls in `torch==2.10`, `torchao==0.15`, `transformers==5.3.0`, the
XNNPACK backend, and the PT2E quantization helpers.

Smoke-test the imports:

```bash
.venv-executorch/bin/python - <<'PY'
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
print("executorch imports ok")
PY
```

> A torchao log line about "Skipping import of cpp extensions due to
> incompatible torch version 2.10.0+cu128" is expected — ExecuTorch's
> Python-only path does not need those C++ kernels.

### 2. Run the int8 export

XNNPACK is CPU-only and the int8 PT2E flow prefers fp32 weights, so the
defaults below are the safe path. The XNNPACK serializer shells out to
`flatc`, which ExecuTorch ships at `.venv-executorch/bin/flatc` — calling
the interpreter directly does **not** activate the venv, so prepend the
bin dir to `PATH` (or `source .venv-executorch/bin/activate` first).

The script automatically neutralizes lerobot's
`smolvlm_with_expert._fp8_quantize_dequantize` round-trip on Q/K/V/probs
(it is there to expose FP8 in the MLIR/IREE flow). ExecuTorch's memory
planner has no size for `float8_e4m3fn`, so without the patch the lower
step crashes with `KeyError: torch.float8_e4m3fn` — see the
`_disable_smolvla_fp8_roundtrip()` helper at the top of
`scripts/export_executorch.py`.

This is the exact command we used to produce
`reports/smolvla_executorch/smolvla_one_step_int8.pte`:

```bash
PATH="$(pwd)/.venv-executorch/bin:$PATH" \
.venv-executorch/bin/python scripts/export_executorch.py \
  --model-id lerobot/smolvla_base \
  --load-device cpu \
  --export-device cpu \
  --export-dtype fp32 \
  --calibration-iters 1 \
  --image-h 128 --image-w 128 --prompt-len 4 \
  --out reports/smolvla_executorch/smolvla_one_step_int8.pte
```

Smaller `--image-h/w` and `--prompt-len` are not about correctness — they
just cap the activation tensor sizes during XNNPACK lowering and
flatbuffer serialization. SmolVLA is ~3.5 B parameters and the lowering
peak can blow past 60 GB resident; the values above were sufficient on a
128 GiB host.

What the script does:

1. Loads `lerobot/smolvla_base` (CUDA is fine for the HF download/init).
2. Builds dummy processed inputs and casts both the model and the inputs
   to CPU + fp32 for `torch.export.export(...)`.
3. Wires up `XNNPACKQuantizer` (per-channel symmetric int8 by default),
   runs `prepare_pt2e`, calibrates with `--calibration-iters` dummy
   samples, then runs `convert_pt2e`.
4. Re-exports the quantized graph and lowers it through
   `to_edge_transform_and_lower([XnnpackPartitioner()]).to_executorch()`.
5. Writes the `.pte` to `--out`.

Useful flags:

- `--no-quant` — skip PT2E and lower the fp32 graph as-is (handy for
  validating that the lowering path works before paying the PT2E cost).
- `--per-tensor` — per-tensor instead of per-channel weight quantization.
- `--dynamic` — dynamic activation quantization (no calibration).
- `--calibration-iters N` — number of dummy samples for static
  activation calibration.
- `--load-device cpu` — use if CUDA is unavailable (slower load).

### 3. Expected output

```
reports/smolvla_executorch/smolvla_one_step_int8.pte   # ~1.5 GiB
```

The first 4 bytes are the size header and bytes 4..8 spell `ET12` (the
canonical ExecuTorch flatbuffer magic). The file is self-contained and
can be loaded by any ExecuTorch runtime build that includes the XNNPACK
backend.

> If the int8 run OOMs on your machine, drop `--calibration-iters` to 1,
> shrink `--image-h/w` and `--prompt-len`, and retry. The XNNPACK
> partitioner + flatbuffer serializer is the memory peak — both scale
> with the activation tensor sizes you exported with.

## PI0 Model

This is the structure of the PI0Policy:

```python
PI0Policy(
  (model): PI0Pytorch(
    (paligemma_with_expert): PaliGemmaWithExpertModel(
      (paligemma): PaliGemmaForConditionalGeneration(
        (model): PaliGemmaModel(
          (vision_tower): SiglipVisionModel(
            (vision_model): SiglipVisionTransformer(
              (embeddings): SiglipVisionEmbeddings(
                (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
                (position_embedding): Embedding(256, 1152)
              )
              (encoder): SiglipEncoder(
                (layers): ModuleList(
                  (0-26): 27 x SiglipEncoderLayer(
                    (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                    (self_attn): SiglipAttention(
                      (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                      (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                      (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                      (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
                    )
                    (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                    (mlp): SiglipMLP(
                      (activation_fn): PytorchGELUTanh()
                      (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                      (fc2): Linear(in_features=4304, out_features=1152, bias=True)
                    )
                  )
                )
              )
              (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
            )
          )
          (multi_modal_projector): PaliGemmaMultiModalProjector(
            (linear): Linear(in_features=1152, out_features=2048, bias=True)
          )
          (language_model): GemmaModel(
            (embed_tokens): Embedding(257152, 2048, padding_idx=0)
            (layers): ModuleList(
              (0-17): 18 x GemmaDecoderLayer(
                (self_attn): GemmaAttention(
                  (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
                  (k_proj): Linear(in_features=2048, out_features=256, bias=False)
                  (v_proj): Linear(in_features=2048, out_features=256, bias=False)
                  (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
                )
                (mlp): GemmaMLP(
                  (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
                  (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
                  (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
                  (act_fn): PytorchGELUTanh()
                )
                (input_layernorm): GemmaRMSNorm((2048,), eps=1e-06)
                (post_attention_layernorm): GemmaRMSNorm((2048,), eps=1e-06)
              )
            )
            (norm): GemmaRMSNorm((2048,), eps=1e-06)
            (rotary_emb): GemmaRotaryEmbedding()
          )
        )
        (lm_head): Linear(in_features=2048, out_features=257152, bias=False)
      )
      (gemma_expert): GemmaForCausalLM(
        (model): GemmaModel(
          (embed_tokens): None
          (layers): ModuleList(
            (0-17): 18 x GemmaDecoderLayer(
              (self_attn): GemmaAttention(
                (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
                (k_proj): Linear(in_features=1024, out_features=256, bias=False)
                (v_proj): Linear(in_features=1024, out_features=256, bias=False)
                (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
              )
              (mlp): GemmaMLP(
                (gate_proj): Linear(in_features=1024, out_features=4096, bias=False)
                (up_proj): Linear(in_features=1024, out_features=4096, bias=False)
                (down_proj): Linear(in_features=4096, out_features=1024, bias=False)
                (act_fn): PytorchGELUTanh()
              )
              (input_layernorm): GemmaRMSNorm((1024,), eps=1e-06)
              (post_attention_layernorm): GemmaRMSNorm((1024,), eps=1e-06)
            )
          )
          (norm): GemmaRMSNorm((1024,), eps=1e-06)
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (lm_head): Linear(in_features=1024, out_features=257152, bias=False)
      )
    )
    (action_in_proj): Linear(in_features=16, out_features=1024, bias=True)
    (action_out_proj): Linear(in_features=1024, out_features=16, bias=True)
    (state_proj): Linear(in_features=32, out_features=1024, bias=True)
    (action_time_mlp_in): Linear(in_features=2048, out_features=1024, bias=True)
    (action_time_mlp_out): Linear(in_features=1024, out_features=1024, bias=True)
  )
)
```

The inference can be separated into three parts. The `vision_tower` handles the encoding of image from the input RGB space into token space. `language_model` contains the main VLM model, and the `gemma_expert` is the smaller action expert that generates the target position using flow-matching process.

Total model parameter is 3,501,339,392 (3.50 B), in which the SigLIP vision model accounts for 412,442,352 (412.44 M), Gemma language model takes up 2,508,531,712 (2508.53 M), and the flow-matching action expert model accounts for 574,788,608 (574.79 M) parameters.

The total required amount of FLOPs is 4,354,614,038,072 (4.35 T) for one inference step.
