from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from understanding_pi0.common.env import print_runtime_info, seed_all, warn_if_mx_execution_unavailable
from understanding_pi0.common.torchao_utils import safe_quantize_linears_, summarize_results
from understanding_pi0.smolvla_mx.loader import (
    build_dummy_processed_inputs,
    load_smolvla_policy,
    print_policy_summary,
)
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
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--out", default="reports/smolvla_mx/smolvla_mx_quantized.pt")
    ap.add_argument("--report-json", default="reports/smolvla_mx/quant_report.json")
    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.device)
    warn_if_mx_execution_unavailable(args.device)

    policy = load_smolvla_policy(args.model_id, device=args.device)
    policy = policy.to(torch.bfloat16)
    print_policy_summary(policy)

    plan = build_quant_plan(policy, quantize_vision=not args.no_vision)
    results = safe_quantize_linears_(
        policy,
        plan=plan,
        quant_device=args.device,
        verbose=True,
    )
    summary = summarize_results(results)

    print("\n[summary]")
    print(f"  applied_mx   : {summary['mx']}")
    print(f"  applied_int8 : {summary['int8']}")
    print(f"  skipped      : {summary['skipped']}")
    print(f"  failed       : {summary['failed']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "state_dict": policy.state_dict(),
            "results": [r.__dict__ for r in results],
        },
        out_path,
    )
    print(f"\n[saved] checkpoint payload -> {out_path}")

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    print(f"[saved] report json -> {report_path}")

    if args.smoke_test:
        sample = build_dummy_processed_inputs(
            policy,
            batch_size=args.batch_size,
            image_hw=(args.image_h, args.image_w),
            prompt_len=args.prompt_len,
            device=args.device,
        )
        with torch.no_grad():
            y = one_step_no_cache(
                policy,
                images=sample["images"],
                img_masks=sample["img_masks"],
                lang_tokens=sample["lang_tokens"],
                lang_masks=sample["lang_masks"],
                state=sample["state"],
                noisy_actions=sample["noisy_actions"],
                timestep=sample["timestep"],
            )
        print(f"[smoke-test] output shape = {tuple(y.shape)}")


if __name__ == "__main__":
    main()