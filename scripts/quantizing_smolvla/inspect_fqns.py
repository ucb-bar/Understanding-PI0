from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from understanding_pi0.common.env import print_runtime_info
from understanding_pi0.smolvla_mx.loader import load_smolvla_policy, print_policy_summary
from understanding_pi0.smolvla_mx.quant_recipe import build_quant_plan, bucket_for_fqn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-vision", action="store_true")
    args = ap.parse_args()

    print_runtime_info(args.device)
    policy = load_smolvla_policy(args.model_id, device=args.device)
    print_policy_summary(policy)

    plan = build_quant_plan(policy, quantize_vision=not args.no_vision)

    print("\n[linears]")
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            print(
                f"{name:120s} "
                f"shape={tuple(mod.weight.shape)!s:22s} "
                f"dtype={str(mod.weight.dtype):12s} "
                f"bucket={bucket_for_fqn(name):14s} "
                f"plan={plan.get(name)}"
            )


if __name__ == "__main__":
    main()