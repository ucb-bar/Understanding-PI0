"""Per-tensor FP8 E4M3 quantization with power-of-two scaling for export.

Implements the pi0-quant / CompGen FP8 po2 quantization scheme as a
self-contained module with no external dependencies beyond PyTorch.

Key difference from MX quantization (``torchao.prototype.mx_formats``):
  - **Per-tensor** scale (single scalar per weight matrix) instead of
    per-block (one scale per 32-element block).
  - No ``in_features % block_size == 0`` constraint -- every linear can be
    quantized, including SmolVLA's Gemma expert with ``hidden_dim=720``.

Usage::

    policy = load_smolvla_policy(...)
    policy, records = apply_fp8_po2_quantization(policy)
    # All linears wrapped in ExportableFP8Po2Linear (fp8 weight + fp8 activation q/dq)
    # RMSNorm patched to bf16 (no f32 upcast)
    # Ready for iree.turbine.aot.export()

Note: f32 elimination in attention/RoPE is done by direct source edits in
``smolvlm_with_expert.py``, NOT by monkey-patching here.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FP8 E4M3 constants
# ---------------------------------------------------------------------------

FP8_E4M3_MAX: float = 448.0
"""Largest finite value representable in float8_e4m3fn (1.75 * 2^8)."""

FP8_E4M3_MAX_PO2: float = 256.0
"""Largest power-of-two representable in float8_e4m3fn (2^8)."""

FP8_E4M3_DTYPE: torch.dtype = torch.float8_e4m3fn
"""PyTorch dtype for E4M3 (4-bit exponent, 3-bit mantissa, no Inf)."""


# ---------------------------------------------------------------------------
# Core quantization primitives (from pi0-quant / CompGen fp8_ops.py)
# ---------------------------------------------------------------------------


def fp8_po2_scale(x: torch.Tensor) -> float:
    """Compute a per-tensor power-of-two scale for FP8 E4M3 quantization.

    ``scale = 2^floor(log2(amax / 256))``.  Fits in NPU E8M0 registers
    (pure exponent shift, no FP multiply).

    Uses ``.item()`` to extract Python scalar before branching so the
    function is safe to call inside ``torch.export`` traces.
    """
    amax_val = x.float().abs().max().item()
    if amax_val == 0:
        return 1.0
    raw_scale = amax_val / FP8_E4M3_MAX_PO2
    if raw_scale < 1.0:
        return 1.0
    return 2.0 ** math.floor(math.log2(raw_scale))


def quantize_weight_fp8_po2(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Quantize a weight tensor to FP8 E4M3 with per-tensor po2 scaling."""
    w_f32 = weight.float()
    scale = fp8_po2_scale(w_f32)
    w_scaled = (w_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(FP8_E4M3_DTYPE)
    return w_fp8, scale


def _fp8_quantize_dequantize(
    x: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Quantize tensor to FP8 and immediately dequantize back.

    Makes the FP8 cast visible in the exported graph.  The fp8 round-trip
    is LOSSY (8-bit precision) and cannot be optimized away by the compiler,
    ensuring fp8 type conversions remain visible at matmul inputs in MLIR.

    Uses unit scale (1.0) for export compatibility — dynamic po2 scale
    computation involves data-dependent branching that breaks torch.export.
    The actual po2 scale is a runtime concern handled by the NPU hardware.
    """
    x_bf16 = x.to(torch.bfloat16)
    x_fp8 = x_bf16.to(FP8_E4M3_DTYPE)
    return x_fp8.to(target_dtype)


# ---------------------------------------------------------------------------
# Export-ready linear wrapper
# ---------------------------------------------------------------------------


class ExportableFP8Po2Linear(nn.Module):
    """Export-friendly FP8 linear with fp8 weight AND fp8 activation q/dq.

    Forward:
      1. Dequantize weight: fp8 -> f32 * scale -> bf16
      2. Quantize activation to fp8 -> dequantize to bf16
      3. Matmul: bf16 x bf16 -> bf16 (both inputs went through fp8)
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: float,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.weight_scale = float(weight_scale)

        self.register_buffer("weight_fp8", weight_fp8, persistent=True)
        if bias is None:
            self.register_buffer("bias_buf", None, persistent=True)
        else:
            self.register_buffer(
                "bias_buf", bias.detach().to(compute_dtype), persistent=True
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> ExportableFP8Po2Linear:
        """Create from a plain nn.Linear by quantizing its weight to FP8 po2."""
        w_fp8, scale = quantize_weight_fp8_po2(linear.weight.data)
        bias = (
            linear.bias.detach().to(compute_dtype)
            if linear.bias is not None
            else None
        )
        return cls(
            weight_fp8=w_fp8,
            weight_scale=scale,
            bias=bias,
            out_features=linear.out_features,
            in_features=linear.in_features,
            compute_dtype=compute_dtype,
        )

    @property
    def weight(self) -> torch.Tensor:
        if self.weight_scale == 1.0:
            return self.weight_fp8.to(self.compute_dtype)
        return (self.weight_fp8.to(torch.float32) * self.weight_scale).to(
            self.compute_dtype
        )

    @property
    def bias(self) -> torch.Tensor | None:
        return self.bias_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight: fp8 -> bf16 (direct, no f32 intermediate)
        w_hp = self.weight_fp8.to(self.compute_dtype)
        if self.weight_scale != 1.0:
            w_hp = w_hp * self.weight_scale
        # Activation: bf16 -> fp8 -> bf16 (fp8 visible in graph)
        x_hp = _fp8_quantize_dequantize(x, target_dtype=self.compute_dtype)
        # Matmul in bf16
        return F.linear(x_hp, w_hp, self.bias_buf)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, "
            f"weight=FP8_E4M3_po2(scale={self.weight_scale})"
        )


# ---------------------------------------------------------------------------
# bf16 precision fix for HuggingFace RMSNorm
# ---------------------------------------------------------------------------


class ExportableFP8Po2Conv2d(nn.Module):
    """Export-friendly FP8 Conv2d with fp8 weight and fp8 activation q/dq."""

    def __init__(self, weight_fp8, bias, stride, padding, dilation, groups,
                 compute_dtype=torch.bfloat16):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.compute_dtype = compute_dtype
        self.register_buffer("weight_fp8", weight_fp8, persistent=True)
        if bias is not None:
            self.register_buffer("bias_buf", bias.to(compute_dtype), persistent=True)
        else:
            self.register_buffer("bias_buf", None, persistent=True)

    @classmethod
    def from_conv2d(cls, conv, compute_dtype=torch.bfloat16):
        w_bf16 = conv.weight.data.to(torch.bfloat16)
        w_fp8 = w_bf16.to(FP8_E4M3_DTYPE)
        bias = conv.bias.detach() if conv.bias is not None else None
        return cls(w_fp8, bias, conv.stride, conv.padding,
                   conv.dilation, conv.groups, compute_dtype)

    def forward(self, x):
        w_hp = self.weight_fp8.to(self.compute_dtype)
        x_hp = _fp8_quantize_dequantize(x, self.compute_dtype)
        return F.conv2d(x_hp, w_hp, self.bias_buf,
                        self.stride, self.padding, self.dilation, self.groups)


def _quantize_conv2d_modules(model, compute_dtype, verbose):
    """Replace nn.Conv2d with ExportableFP8Po2Conv2d."""
    count = 0
    for parent, child_name, child, fqn in _iter_named_children_with_parent(model):
        if isinstance(child, nn.Conv2d):
            new_mod = ExportableFP8Po2Conv2d.from_conv2d(child, compute_dtype)
            setattr(parent, child_name, new_mod)
            count += 1
            if verbose:
                print(f"[fp8_po2] conv2d {fqn}: {tuple(new_mod.weight_fp8.shape)}")
    if verbose and count:
        print(f"[fp8_po2] quantized {count} Conv2d modules")
    return count


def _patch_norm_modules(model: nn.Module, verbose: bool) -> int:
    """Replace RMSNorm.forward to compute in input dtype instead of f32.

    Instance-level method replacement — targets specific module instances
    by type, not global function names.
    """
    count = 0
    for name, mod in model.named_modules():
        cls_name = type(mod).__name__
        if "RMSNorm" not in cls_name:
            continue
        if not hasattr(mod, "weight"):
            continue

        eps = getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6))
        w = mod.weight

        def _make_forward(_eps, _w):
            def forward(x: torch.Tensor) -> torch.Tensor:
                var = x.pow(2).mean(-1, keepdim=True)
                normed = x * torch.rsqrt(var + _eps)
                return normed * (1.0 + _w.to(x.dtype))

            return forward

        mod.forward = _make_forward(eps, w)
        count += 1

    if verbose and count:
        print(f"[bf16_precision] patched {count} RMSNorm modules (bf16)")
    return count


# ---------------------------------------------------------------------------
# Quantization plan + apply
# ---------------------------------------------------------------------------


@dataclass
class FP8Po2Record:
    """Record of what happened to each linear during quantization."""

    fqn: str
    replaced: bool
    reason: str


def build_fp8_po2_plan(
    model: nn.Module,
    quantize_vision: bool = True,
) -> OrderedDict[str, str | None]:
    """Build a quantization plan that assigns fp8_po2 to ALL linears.

    No block_size constraint, no int8 fallback.
    """
    plan: OrderedDict[str, str | None] = OrderedDict()

    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        if "lm_head" in fqn:
            plan[fqn] = None
            continue

        if not quantize_vision and (
            "vision_model" in fqn or ".connector." in fqn
        ):
            plan[fqn] = None
            continue

        plan[fqn] = "fp8_po2"

    return plan


def apply_fp8_po2_quantization(
    model: nn.Module,
    quantize_vision: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> tuple[nn.Module, list[FP8Po2Record]]:
    """Quantize all linears to FP8 po2 + patch RMSNorm for bf16.

    Steps:
      1. Deep-copy the model.
      2. Replace all nn.Linear with ExportableFP8Po2Linear (fp8 weights +
         fp8 activation q/dq at each matmul input).
      3. Patch RMSNorm to compute in bf16 (not f32).

    Note: attention f32 and RoPE f32 are fixed at the source in
    ``smolvlm_with_expert.py``, not patched here.
    """
    plan = build_fp8_po2_plan(model, quantize_vision=quantize_vision)

    model_copy = copy.deepcopy(model)

    # --- Step 1: Replace linears with FP8 po2 wrappers ---
    records: list[FP8Po2Record] = []

    targets: list[tuple[nn.Module, str, nn.Module, str]] = []
    for parent, child_name, child, fqn in _iter_named_children_with_parent(
        model_copy
    ):
        if (
            isinstance(child, nn.Linear)
            and fqn in plan
            and plan[fqn] == "fp8_po2"
        ):
            targets.append((parent, child_name, child, fqn))

    for parent, child_name, child, fqn in targets:
        try:
            new_mod = ExportableFP8Po2Linear.from_linear(
                child, compute_dtype=compute_dtype
            )
            setattr(parent, child_name, new_mod)
            records.append(
                FP8Po2Record(fqn=fqn, replaced=True, reason="fp8_po2")
            )
            if verbose:
                w = new_mod.weight_fp8
                print(
                    f"[fp8_po2] {fqn}: "
                    f"{tuple(w.shape)} scale={new_mod.weight_scale}"
                )
        except Exception as e:
            records.append(
                FP8Po2Record(
                    fqn=fqn,
                    replaced=False,
                    reason=f"{type(e).__name__}: {e}",
                )
            )
            if verbose:
                print(f"[fp8_po2] FAILED {fqn}: {type(e).__name__}: {e}")

    for fqn, action in plan.items():
        if action is None:
            records.append(
                FP8Po2Record(fqn=fqn, replaced=False, reason="skipped")
            )

    # --- Step 2: Quantize Conv2d weights to fp8 ---
    conv_count = _quantize_conv2d_modules(model_copy, compute_dtype, verbose)

    # --- Step 3: Patch RMSNorm (HuggingFace code we don't control) ---
    _patch_norm_modules(model_copy, verbose=verbose)

    n_replaced = sum(1 for r in records if r.replaced)
    n_skipped = sum(1 for r in records if r.reason == "skipped")
    n_failed = sum(
        1 for r in records if not r.replaced and r.reason != "skipped"
    )
    if verbose:
        print(
            f"[fp8_po2] done: {n_replaced} replaced, "
            f"{n_skipped} skipped, {n_failed} failed"
        )

    return model_copy, records


def summarize_fp8_po2_records(
    records: list[FP8Po2Record],
) -> dict[str, int]:
    """Summarize quantization records."""
    return {
        "fp8_po2": sum(1 for r in records if r.replaced),
        "skipped": sum(1 for r in records if r.reason == "skipped"),
        "failed": sum(
            1 for r in records if not r.replaced and r.reason != "skipped"
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_named_children_with_parent(module: nn.Module, prefix: str = ""):
    """Yield (parent, child_name, child, fqn) for every module in the tree."""
    for child_name, child in module.named_children():
        fqn = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, child, fqn
        yield from _iter_named_children_with_parent(child, fqn)
