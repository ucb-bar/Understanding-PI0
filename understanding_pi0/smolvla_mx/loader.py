from __future__ import annotations

from typing import Any

import torch

from understanding_pi0.common.hf_compat import patch_diffusers_torchao_logger_bug


def _import_lerobot_smolvla():
    # Patch diffusers BEFORE importing lerobot.
    patch_diffusers_torchao_logger_bug(verbose=True)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    return SmolVLAPolicy, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def load_smolvla_policy(
    model_id: str = "lerobot/smolvla_base",
    device: str = "cuda",
):
    SmolVLAPolicy, _, _, _ = _import_lerobot_smolvla()
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    if hasattr(policy.config, "compile_model"):
        policy.config.compile_model = False
    return policy


def print_policy_summary(policy) -> None:
    cfg = policy.config
    print("[smolvla]")
    print(f"  chunk_size               : {cfg.chunk_size}")
    print(f"  n_action_steps           : {cfg.n_action_steps}")
    print(f"  num_steps                : {cfg.num_steps}")
    print(f"  num_vlm_layers           : {cfg.num_vlm_layers}")
    print(f"  expert_width_multiplier  : {cfg.expert_width_multiplier}")
    print(f"  attention_mode           : {cfg.attention_mode}")
    print(f"  resize_imgs_with_padding : {cfg.resize_imgs_with_padding}")
    print(f"  tokenizer_max_length     : {cfg.tokenizer_max_length}")


def build_dummy_raw_batch(
    policy,
    batch_size: int = 1,
    image_hw: tuple[int, int] = (256, 256),
    prompt_len: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    _, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE = _import_lerobot_smolvla()

    h, w = image_hw
    cfg = policy.config

    image_keys = list(cfg.image_features.keys())
    if not image_keys:
        raise RuntimeError("policy.config.image_features is empty")

    tok_len = int(cfg.tokenizer_max_length)
    prompt_len = max(0, min(prompt_len, tok_len))

    batch: dict[str, Any] = {}

    for key in image_keys:
        batch[key] = torch.rand(batch_size, 3, h, w, device=device, dtype=torch.bfloat16)

    batch[OBS_STATE] = torch.randn(
        batch_size,
        int(cfg.max_state_dim),
        device=device,
        dtype=torch.float32,
    )

    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
        batch_size,
        tok_len,
        device=device,
        dtype=torch.long,
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros(
        batch_size,
        tok_len,
        device=device,
        dtype=torch.bool,
    )

    if prompt_len > 0:
        batch[OBS_LANGUAGE_TOKENS][:, :prompt_len] = 1
        batch[OBS_LANGUAGE_ATTENTION_MASK][:, :prompt_len] = True

    if hasattr(policy, "_prepare_batch"):
        batch = policy._prepare_batch(batch)

    return batch


def build_dummy_processed_inputs(
    policy,
    batch_size: int = 1,
    image_hw: tuple[int, int] = (256, 256),
    prompt_len: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    _, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, _ = _import_lerobot_smolvla()

    batch = build_dummy_raw_batch(
        policy=policy,
        batch_size=batch_size,
        image_hw=image_hw,
        prompt_len=prompt_len,
        device=device,
    )

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    noisy_actions = torch.randn(
        batch_size,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
        device=device,
        dtype=torch.float32,
    )
    timestep = torch.full(
        (batch_size,),
        0.5,
        device=device,
        dtype=torch.float32,
    )

    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "noisy_actions": noisy_actions,
        "timestep": timestep,
    }