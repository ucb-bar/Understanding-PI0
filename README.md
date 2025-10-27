# Understanding PI0

This repository contains some basic scripts to understand the [PI0 policy](https://github.com/Physical-Intelligence/openpi).


## Setting up the repository

We use the Huggingface implementation of PI0 policy. In order to look into the source code, it is recommended to pull the lerobot and transformers source code.

```bash
git clone https://github.com/huggingface/lerobot.git
```

Note that to use transformers with lerobot, we need to checkout to this specific branch:

```bash
git clone https://github.com/huggingface/transformers.git
cd ./transformers/
git checkout fix/lerobot_openpi
```


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



