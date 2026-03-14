from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from understanding_pi0.common.env import (
    print_runtime_info,
    seed_all,
    warn_if_mx_execution_unavailable,
)
from understanding_pi0.common.iree_ocp_patch import apply_all_iree_ocp_patches
from understanding_pi0.common.mx_exportable import clone_and_rewrite_quantized_linears_for_export
from understanding_pi0.common.torchao_utils import safe_quantize_linears_
from understanding_pi0.smolvla_mx.loader import (
    build_dummy_processed_inputs,
    load_smolvla_policy,
)
from understanding_pi0.smolvla_mx.quant_recipe import build_quant_plan
from understanding_pi0.smolvla_mx.wrappers import (
    SmolVLAOneStepNoCacheWrapper,
    flatten_processed_inputs,
)


def save_mlir_fallback(exported, out_path: Path) -> None:
    """
    Handle minor IREE Turbine API differences across versions.
    """
    if hasattr(exported, "save_mlir"):
        exported.save_mlir(str(out_path))
        return

    if hasattr(exported, "mlir_module"):
        out_path.write_text(str(exported.mlir_module))
        return

    if hasattr(exported, "module"):
        out_path.write_text(str(exported.module))
        return

    raise RuntimeError("Could not find a way to save MLIR from the exported object.")


def compile_vmfb_fallback(exported, vmfb_path: Path) -> None:
    """
    Handle minor IREE Turbine compile API differences across versions.
    """
    try:
        exported.compile(save_to=str(vmfb_path))
        return
    except TypeError:
        pass

    compiled = exported.compile(save_to=None)
    if isinstance(compiled, (bytes, bytearray)):
        vmfb_path.write_bytes(bytes(compiled))
        return

    try:
        vmfb_path.write_bytes(bytes(compiled))
        return
    except Exception as e:
        raise RuntimeError("Could not save VMFB from exported.compile(...) result") from e


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
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--no-exportable-mx", action="store_true")
    ap.add_argument(
        "--mx-kernel-preference",
        default="AUTO",
        help="KernelPreference for MX quantization (for example: AUTO or EMULATED).",
    )
    ap.add_argument("--skip-patches", action="store_true")

    ap.add_argument("--print-readable", action="store_true")
    ap.add_argument("--compile-vmfb", action="store_true")

    ap.add_argument("--out", default="reports/smolvla_mx/smolvla_one_step.mlir")
    ap.add_argument("--vmfb-out", default="reports/smolvla_mx/smolvla_one_step.vmfb")
    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.device)
    warn_if_mx_execution_unavailable(args.device)

    if not args.skip_patches:
        apply_all_iree_ocp_patches(verbose=True)

    import iree.turbine.aot as aot

    # Load baseline policy
    policy = load_smolvla_policy(args.model_id, device=args.device).to(torch.bfloat16).eval()

    # Apply quantization recipe
    if not args.no_quant:
        plan = build_quant_plan(policy, quantize_vision=not args.no_vision)
        _ = safe_quantize_linears_(
            policy,
            plan=plan,
            quant_device=args.device,
            mx_kernel_preference=args.mx_kernel_preference,
            verbose=False,
        )

    # Rewrite MXTensor-backed linears into exportable wrappers that:
    #   - keep MX storage
    #   - explicitly dequantize inside forward
    # This is required on non-SM100 hardware to avoid MXTensor eager dispatch
    # failures such as aten.expand.
    if not args.no_exportable_mx:
        policy, records = clone_and_rewrite_quantized_linears_for_export(
            policy,
            compute_dtype=torch.bfloat16,
            verbose=False,
        )
        n_replaced = sum(int(r.replaced) for r in records)
        print(f"[exportable_linear] replaced {n_replaced} MX linears for export")

    # Build processed example inputs
    sample = build_dummy_processed_inputs(
        policy,
        batch_size=args.batch_size,
        image_hw=(args.image_h, args.image_w),
        prompt_len=args.prompt_len,
        device=args.device,
    )

    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=len(sample["images"])).eval()
    example_args = flatten_processed_inputs(sample)

    print("[export] running iree.turbine.aot.export(...)")
    exported = aot.export(
        wrapper,
        args=example_args,
        strict_export=False,
    )

    if args.print_readable and hasattr(exported, "print_readable"):
        print("\n[readable ir]")
        exported.print_readable()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_mlir_fallback(exported, out_path)
    print(f"[saved] mlir -> {out_path}")

    if args.compile_vmfb:
        vmfb_path = Path(args.vmfb_out)
        vmfb_path.parent.mkdir(parents=True, exist_ok=True)
        print("[compile] running exported.compile(...)")
        compile_vmfb_fallback(exported, vmfb_path)
        print(f"[saved] vmfb -> {vmfb_path}")


if __name__ == "__main__":
    main()
