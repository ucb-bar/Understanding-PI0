from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def build_quant_plan(
    model,
    quantize_vision: bool = True,
    mx_block_size: int = 32,
) -> OrderedDict[str, str | None]:
    """
    MX-first plan:
      - use MX only when in_features is divisible by mx_block_size
      - otherwise use int8 directly
      - skip lm_head
      - optionally skip vision path
    """
    plan: OrderedDict[str, str | None] = OrderedDict()

    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        if "lm_head" in fqn:
            plan[fqn] = None
            continue

        if not quantize_vision and ("vision_model" in fqn or ".connector." in fqn):
            plan[fqn] = None
            continue

        in_features = int(mod.weight.shape[-1])

        if in_features % mx_block_size == 0:
            plan[fqn] = "mx"
        else:
            plan[fqn] = "int8"

    return plan


def bucket_for_fqn(fqn: str) -> str:
    if ".vision_model." in fqn:
        return "vision"
    if ".connector." in fqn:
        return "connector"
    if ".text_model.layers." in fqn:
        return "vlm_text"
    if ".lm_expert.layers." in fqn:
        return "expert"
    if "state_proj" in fqn:
        return "state_proj"
    if "action_in_proj" in fqn:
        return "action_in_proj"
    if "action_out_proj" in fqn:
        return "action_out_proj"
    if "action_time_mlp_" in fqn:
        return "action_time"
    return "other"