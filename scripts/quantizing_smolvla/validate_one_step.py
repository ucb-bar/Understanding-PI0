from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from understanding_pi0.common.env import print_runtime_info, seed_all, warn_if_mx_execution_unavailable
from understanding_pi0.common.mx_exportable import clone_and_rewrite_quantized_linears_for_export
from understanding_pi0.common.torchao_utils import safe_quantize_linears_
from understanding_pi0.smolvla_mx.loader import build_dummy_processed_inputs, load_smolvla_policy
from understanding_pi0.smolvla_mx.quant_recipe import build_quant_plan
from understanding_pi0.smolvla_mx.wrappers import one_step_no_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-exportable-mx", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.device)
    warn_if_mx_execution_unavailable(args.device)

    baseline = load_smolvla_policy(args.model_id, device=args.device).to(torch.bfloat16)
    quantized = load_smolvla_policy(args.model_id, device=args.device).to(torch.bfloat16)

    plan = build_quant_plan(quantized, quantize_vision=not args.no_vision)
    _ = safe_quantize_linears_(
        quantized,
        plan=plan,
        quant_device=args.device,
        verbose=False,
    )

    if not args.no_exportable_mx:
        quantized, records = clone_and_rewrite_quantized_linears_for_export(
            quantized, compute_dtype=torch.bfloat16, verbose=False
        )
        n_replaced = sum(int(r.replaced) for r in records)
        print(f"[exportable_linear] replaced {n_replaced} quantized linears for validation")

    sample = build_dummy_processed_inputs(
        baseline,
        batch_size=args.batch_size,
        image_hw=(args.image_h, args.image_w),
        prompt_len=args.prompt_len,
        device=args.device,
    )

    with torch.no_grad():
        y_ref = one_step_no_cache(
            baseline,
            images=sample["images"],
            img_masks=sample["img_masks"],
            lang_tokens=sample["lang_tokens"],
            lang_masks=sample["lang_masks"],
            state=sample["state"],
            noisy_actions=sample["noisy_actions"],
            timestep=sample["timestep"],
        )
        y_q = one_step_no_cache(
            quantized,
            images=sample["images"],
            img_masks=sample["img_masks"],
            lang_tokens=sample["lang_tokens"],
            lang_masks=sample["lang_masks"],
            state=sample["state"],
            noisy_actions=sample["noisy_actions"],
            timestep=sample["timestep"],
        )

    diff = y_q - y_ref
    mse = F.mse_loss(y_q, y_ref).item()
    mean_abs = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    cosine = F.cosine_similarity(y_ref.flatten(1), y_q.flatten(1), dim=1).mean().item()

    print("[validation]")
    print(f"  ref shape  : {tuple(y_ref.shape)}")
    print(f"  quant shape: {tuple(y_q.shape)}")
    print(f"  mse        : {mse:.8f}")
    print(f"  mean_abs   : {mean_abs:.8f}")
    print(f"  max_abs    : {max_abs:.8f}")
    print(f"  cosine     : {cosine:.8f}")


if __name__ == "__main__":
    main()
