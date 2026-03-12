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
from understanding_pi0.common.mx_exportable import clone_and_rewrite_mx_linears_for_export
from understanding_pi0.common.torchao_utils import safe_quantize_linears_
from understanding_pi0.smolvla_mx.loader import build_dummy_processed_inputs, load_smolvla_policy
from understanding_pi0.smolvla_mx.quant_recipe import build_quant_plan
from understanding_pi0.smolvla_mx.wrappers import (
    SmolVLAOneStepNoCacheWrapper,
    flatten_processed_inputs,
)


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
    ap.add_argument("--exportable-mx", action="store_true", default=True)
    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.device)
    warn_if_mx_execution_unavailable(args.device)

    policy = load_smolvla_policy(args.model_id, device=args.device).to(torch.bfloat16)
    plan = build_quant_plan(policy, quantize_vision=not args.no_vision)
    _ = safe_quantize_linears_(policy, plan=plan, quant_device=args.device, verbose=False)

    if args.exportable_mx:
        policy, records = clone_and_rewrite_mx_linears_for_export(
            policy, compute_dtype=torch.bfloat16, verbose=False
        )
        n_replaced = sum(int(r.replaced) for r in records)
        print(f"[mx_exportable] replaced {n_replaced} MX linears for compile check")

    sample = build_dummy_processed_inputs(
        policy,
        batch_size=args.batch_size,
        image_hw=(args.image_h, args.image_w),
        prompt_len=args.prompt_len,
        device=args.device,
    )

    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=len(sample["images"])).eval()
    example_args = flatten_processed_inputs(sample)

    with torch.no_grad():
        y_eager = wrapper(*example_args)

    compiled = torch.compile(wrapper, backend="eager", fullgraph=True)
    with torch.no_grad():
        y_compiled = compiled(*example_args)

    diff = y_compiled - y_eager
    mse = F.mse_loss(y_compiled, y_eager).item()
    mean_abs = diff.abs().mean().item()
    max_abs = diff.abs().max().item()

    print("[compile-check]")
    print(f"  eager shape    : {tuple(y_eager.shape)}")
    print(f"  compiled shape : {tuple(y_compiled.shape)}")
    print(f"  mse            : {mse:.8f}")
    print(f"  mean_abs       : {mean_abs:.8f}")
    print(f"  max_abs        : {max_abs:.8f}")
    print("  status         : torch.compile(backend='eager', fullgraph=True) succeeded")


if __name__ == "__main__":
    main()