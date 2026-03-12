from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def _first_param_dtype(module: nn.Module, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    for p in module.parameters():
        if p.is_floating_point():
            return p.dtype
    for b in module.buffers():
        if b.is_floating_point():
            return b.dtype
    return default


def _maybe_cast_tensor(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_floating_point(x):
        return x.to(dtype)
    return x


def _maybe_cast_list(xs, dtype: torch.dtype):
    out = []
    for x in xs:
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            out.append(x.to(dtype))
        else:
            out.append(x)
    return out


@torch.no_grad()
def one_step_no_cache(
    policy,
    images,
    img_masks,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noisy_actions: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    One export-friendly denoise step without KV-cache filling.
    Output shape:
      [B, chunk_size, max_action_dim]
    """
    model = policy.model

    # Compute dtypes from the actual submodules that consume these tensors.
    vision_dtype = _first_param_dtype(model.vlm_with_expert.vlm.model.vision_model, torch.bfloat16)
    state_dtype = _first_param_dtype(model.state_proj, torch.bfloat16)
    action_dtype = _first_param_dtype(model.action_in_proj, torch.bfloat16)
    time_dtype = _first_param_dtype(model.action_time_mlp_in, action_dtype)
    out_dtype = _first_param_dtype(model.action_out_proj, torch.float32)

    images = _maybe_cast_list(images, vision_dtype)
    state = _maybe_cast_tensor(state, state_dtype)
    noisy_actions = _maybe_cast_tensor(noisy_actions, action_dtype)
    timestep = _maybe_cast_tensor(timestep, time_dtype)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
        noisy_actions, timestep
    )

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    out = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )

    if isinstance(out, tuple) and len(out) == 2:
        outputs_embeds, _ = out
    else:
        outputs_embeds = out

    if isinstance(outputs_embeds, (tuple, list)):
        suffix_out = outputs_embeds[1]
    else:
        raise RuntimeError(f"Unexpected output type from vlm_with_expert.forward: {type(outputs_embeds)}")

    suffix_out = suffix_out[:, -model.config.chunk_size :]
    suffix_out = suffix_out.to(out_dtype if torch.is_floating_point(torch.empty((), dtype=out_dtype)) else torch.float32)
    v_t = model.action_out_proj(suffix_out)
    return v_t


def flatten_processed_inputs(sample: dict) -> tuple:
    flat = []
    flat.extend(sample["images"])
    flat.extend(sample["img_masks"])
    flat.extend(
        [
            sample["lang_tokens"],
            sample["lang_masks"],
            sample["state"],
            sample["noisy_actions"],
            sample["timestep"],
        ]
    )
    return tuple(flat)


class SmolVLAOneStepNoCacheWrapper(nn.Module):
    def __init__(self, policy, num_cams: int):
        super().__init__()
        self.policy = policy.eval()
        self.num_cams = num_cams

    def forward(self, *flat_inputs):
        expected = 2 * self.num_cams + 5
        if len(flat_inputs) != expected:
            raise ValueError(f"expected {expected} flat inputs, got {len(flat_inputs)}")

        images = list(flat_inputs[: self.num_cams])
        img_masks = list(flat_inputs[self.num_cams : 2 * self.num_cams])
        lang_tokens = flat_inputs[2 * self.num_cams + 0]
        lang_masks = flat_inputs[2 * self.num_cams + 1]
        state = flat_inputs[2 * self.num_cams + 2]
        noisy_actions = flat_inputs[2 * self.num_cams + 3]
        timestep = flat_inputs[2 * self.num_cams + 4]

        return one_step_no_cache(
            self.policy,
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noisy_actions=noisy_actions,
            timestep=timestep,
        )