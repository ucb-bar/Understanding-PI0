"""
Export a SmolVLA one-step wrapper to an int8-quantized ExecuTorch ``.pte``.

This mirrors :mod:`scripts.export_iree`, but instead of going through Turbine
and IREE, the model is lowered with ExecuTorch's XNNPACK delegate after
PT2E (post-training, "PyTorch 2 Export") int8 quantization.

The PT2E flow used here is the canonical ExecuTorch path:

    1. ``torch.export.export(model, args).module()`` produces a graph module.
    2. ``prepare_pt2e`` inserts observers using ``XNNPACKQuantizer``.
    3. The prepared module is run on a few calibration samples.
    4. ``convert_pt2e`` rewrites the graph to int8 ops.
    5. The quantized graph is re-exported and lowered with
       ``XnnpackPartitioner`` -> ``to_executorch()`` -> ``.pte``.

Notes on environment
--------------------
ExecuTorch pins ``torchao==0.15`` for 1.1.x while this project pins
``torchao==0.16``. The two extras (``export_iree`` and ``export_executorch``)
are not co-installable in a single resolved environment - install only the
one you need at a time, or use separate virtualenvs.

The XNNPACK delegate runs on CPU and prefers fp32 weights. By default this
script loads the policy on CUDA (matching the IREE path) but moves the
wrapped one-step model to CPU + fp32 before export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from understanding_pi0.common.env import print_runtime_info, seed_all
from understanding_pi0.smolvla_mx.loader import (
    build_dummy_processed_inputs,
    load_smolvla_policy,
)
from understanding_pi0.smolvla_mx.wrappers import (
    SmolVLAOneStepNoCacheWrapper,
    flatten_processed_inputs,
)


def _disable_smolvla_fp8_roundtrip() -> None:
    """
    lerobot's smolvla SDPA inserts ``bf16 -> float8_e4m3fn -> bf16`` round-trips
    on Q/K/V/probs (see ``smolvlm_with_expert._fp8_quantize_dequantize``) to
    make FP8 visible in the MLIR graph for the IREE export flow. ExecuTorch's
    memory planner does not know how to size ``float8_e4m3fn`` activations
    and fails the lowering with ``KeyError: torch.float8_e4m3fn``. For the
    ExecuTorch path we replace the round-trip with the identity so no fp8
    dtype reaches the graph.
    """
    try:
        from lerobot.policies.smolvla import smolvlm_with_expert  # type: ignore
    except Exception as e:
        print(f"[fp8-patch] could not import smolvlm_with_expert: {e}")
        return

    def _identity(x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        return x.to(target_dtype) if x.dtype != target_dtype else x

    smolvlm_with_expert._fp8_quantize_dequantize = _identity
    print("[fp8-patch] _fp8_quantize_dequantize replaced with identity")


def _move_inputs(args_tuple: tuple, device: str, float_dtype: torch.dtype) -> tuple:
    """Move a flat input tuple onto ``device`` and cast floating tensors to ``float_dtype``."""
    out = []
    for x in args_tuple:
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                out.append(x.to(device=device, dtype=float_dtype))
            else:
                out.append(x.to(device=device))
        else:
            out.append(x)
    return tuple(out)


def _build_calibration_inputs(
    policy,
    *,
    num_iters: int,
    batch_size: int,
    image_hw: tuple[int, int],
    prompt_len: int,
    src_device: str,
    export_device: str,
    export_dtype: torch.dtype,
) -> list[tuple]:
    """Build a list of flat input tuples to feed the prepared (observed) model."""
    samples: list[tuple] = []
    for i in range(num_iters):
        sample = build_dummy_processed_inputs(
            policy,
            batch_size=batch_size,
            image_hw=image_hw,
            prompt_len=prompt_len,
            device=src_device,
        )
        flat = flatten_processed_inputs(sample)
        samples.append(_move_inputs(flat, export_device, export_dtype))
        if i == 0:
            print(f"[calib] sample[0] tensor count = {len(flat)}")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument(
        "--load-device",
        default="cuda",
        help="Device for loading the HF policy (CUDA recommended for speed).",
    )
    ap.add_argument(
        "--export-device",
        default="cpu",
        help="Device used for torch.export / PT2E. XNNPACK is CPU-only, keep 'cpu'.",
    )
    ap.add_argument(
        "--export-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Dtype to cast the model to before export. XNNPACK int8 prefers fp32.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=8)

    ap.add_argument(
        "--no-quant",
        action="store_true",
        help="Skip PT2E int8 quantization and lower the fp model directly.",
    )
    ap.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use per-tensor quantization (default: per-channel, recommended for accuracy).",
    )
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic activation quantization (default: static, with calibration).",
    )
    ap.add_argument(
        "--calibration-iters",
        type=int,
        default=4,
        help="Number of dummy samples used to calibrate static activation ranges.",
    )

    ap.add_argument(
        "--out",
        default="reports/smolvla_executorch/smolvla_one_step_int8.pte",
    )

    args = ap.parse_args()

    seed_all(args.seed)
    print_runtime_info(args.load_device)

    _disable_smolvla_fp8_roundtrip()

    export_dtype = torch.float32 if args.export_dtype == "fp32" else torch.bfloat16

    # ------------------------------------------------------------------
    # 1) Load policy and build a CPU-friendly export wrapper.
    # ------------------------------------------------------------------
    print("[load] loading SmolVLA policy")
    policy = load_smolvla_policy(args.model_id, device=args.load_device).eval()

    # We'll generate calibration samples using the loaded policy on its
    # own device (faster on CUDA), then move tensors to the export device.
    sample = build_dummy_processed_inputs(
        policy,
        batch_size=args.batch_size,
        image_hw=(args.image_h, args.image_w),
        prompt_len=args.prompt_len,
        device=args.load_device,
    )
    num_cams = len(sample["images"])
    example_args = _move_inputs(
        flatten_processed_inputs(sample), args.export_device, export_dtype
    )

    # Move the policy to the export device + dtype for tracing / lowering.
    print(f"[load] moving policy to {args.export_device} ({args.export_dtype})")
    policy = policy.to(device=args.export_device, dtype=export_dtype).eval()

    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=num_cams).eval()

    # ------------------------------------------------------------------
    # 2) torch.export the wrapper as a graph module suitable for PT2E.
    # ------------------------------------------------------------------
    print("[export] torch.export.export(wrapper, ...)")
    exported_program = torch.export.export(wrapper, example_args, strict=False)
    graph_module = exported_program.module()

    # ------------------------------------------------------------------
    # 3) Optional: PT2E int8 quantization with the XNNPACK quantizer.
    # ------------------------------------------------------------------
    if not args.no_quant:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        is_per_channel = not args.per_tensor
        qconfig = get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=args.dynamic,
        )
        print(
            f"[pt2e] XNNPACKQuantizer "
            f"(per_channel={is_per_channel}, dynamic={args.dynamic})"
        )
        quantizer = XNNPACKQuantizer().set_global(qconfig)

        prepared = prepare_pt2e(graph_module, quantizer)

        if not args.dynamic:
            calib_samples = _build_calibration_inputs(
                policy,
                num_iters=args.calibration_iters,
                batch_size=args.batch_size,
                image_hw=(args.image_h, args.image_w),
                prompt_len=args.prompt_len,
                src_device=args.load_device,
                export_device=args.export_device,
                export_dtype=export_dtype,
            )
            print(f"[pt2e] calibrating with {len(calib_samples)} samples")
            with torch.no_grad():
                for i, calib_args in enumerate(calib_samples):
                    prepared(*calib_args)
                    print(f"[pt2e]   calib step {i + 1}/{len(calib_samples)} ok")

        print("[pt2e] convert_pt2e -> int8 graph")
        quantized = convert_pt2e(prepared)
    else:
        quantized = graph_module

    # ------------------------------------------------------------------
    # 4) Re-export the (quantized) module and lower to ExecuTorch.
    # ------------------------------------------------------------------
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    print("[export] torch.export.export(quantized, ...)")
    final_ep = torch.export.export(quantized, example_args, strict=False)

    # SmolVLA's preprocessing produces several ops that are not in the
    # Core ATen opset (e.g. `aten.bucketize.Tensor`, `aten.empty_permuted`).
    # XNNPACK does not claim them, but the portable CPU kernels handle
    # them just fine - so we explicitly skip the strict core-ATen
    # verifier rather than chase each operator one by one.
    edge_compile_config = EdgeCompileConfig(
        _check_ir_validity=False,
    )

    print("[lower] to_edge_transform_and_lower([XnnpackPartitioner()])")
    edge_program = to_edge_transform_and_lower(
        final_ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=edge_compile_config,
    )

    print("[lower] to_executorch()")
    et_program = edge_program.to_executorch()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(et_program, "save_to_file"):
        et_program.save_to_file(str(out_path))
    else:
        # Older API: .buffer is a bytes-like object holding the .pte payload.
        out_path.write_bytes(bytes(et_program.buffer))
    print(f"[saved] pte -> {out_path}")


if __name__ == "__main__":
    main()
