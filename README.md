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
