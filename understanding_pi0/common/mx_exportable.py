from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.mx_formats.constants import (
    E8M0_EXPONENT_NAN_VAL,
    F32_MIN_NORMAL,
)
from torchao.prototype.mx_formats.utils import from_blocked

MBITS_F32 = 23


def is_mx_tensor(x) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and hasattr(x, "qdata")
        and hasattr(x, "scale")
        and hasattr(x, "elem_dtype")
        and hasattr(x, "block_size")
        and x.__class__.__name__ == "MXTensor"
    )


def is_other_quantized_tensor(x) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if is_mx_tensor(x):
        return False
    # TorchAO tensor subclasses used by int8/intx paths generally expose dequantize().
    if hasattr(x, "dequantize"):
        return True
    # Extra belt-and-suspenders for class-name based detection.
    cls = x.__class__.__name__
    return cls in {
        "AffineQuantizedTensor",
        "LinearActivationQuantizedTensor",
    }


@dataclass
class ReplaceRecord:
    fqn: str
    replaced: bool
    reason: str


def _decode_e8m0_scale_to_fp(scale_e8m0: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    scale_u8 = scale_e8m0.view(torch.uint8)
    scale_i32 = scale_u8.to(torch.int32)

    scale_fp32_bits = torch.bitwise_left_shift(scale_i32, MBITS_F32)
    scale_fp32 = scale_fp32_bits.view(torch.float32)

    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    scale_fp32 = torch.where(
        scale_u8 != E8M0_EXPONENT_NAN_VAL,
        scale_fp32,
        torch.full_like(scale_fp32, float("nan"), dtype=torch.float32),
    )

    return scale_fp32.to(target_dtype)


def _manual_dequantize_mx_tensor(
    mx: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert is_mx_tensor(mx), f"expected MXTensor, got {type(mx)}"

    qdata = mx.qdata
    scale = mx.scale
    elem_dtype = mx.elem_dtype
    block_size = int(mx.block_size)
    is_swizzled_scales = bool(getattr(mx, "is_swizzled_scales", False))

    if elem_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            f"manual MX export path currently supports only MXFP8 qdata, got {elem_dtype}"
        )

    orig_shape = tuple(qdata.shape)
    is_transposed = not qdata.is_contiguous()

    if is_transposed:
        qdata = qdata.t()
        orig_shape = (orig_shape[1], orig_shape[0])

    assert qdata.is_contiguous(), "expected contiguous qdata after transpose normalization"
    M, K = orig_shape
    assert K % block_size == 0, f"last dim {K} must be divisible by block_size {block_size}"

    if is_swizzled_scales:
        scale = from_blocked(scale.flatten(), M, K // block_size)
    else:
        scale = scale.view(M, K // block_size)

    data_hp = qdata.to(target_dtype)
    s_fp = _decode_e8m0_scale_to_fp(scale, target_dtype)

    data_hp = data_hp.view(M, K // block_size, block_size)
    s_fp = s_fp.unsqueeze(-1)
    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(M, K)

    if is_transposed:
        data_hp = data_hp.t()

    return data_hp


class ExportableMXLinear(nn.Module):
    def __init__(
        self,
        weight_mx: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype

        self.register_buffer("weight_mx", weight_mx, persistent=True)
        if bias is None:
            self.register_buffer("bias_buf", None, persistent=True)
        else:
            self.register_buffer("bias_buf", bias.detach(), persistent=True)

    @classmethod
    def from_linear(cls, linear: nn.Linear, compute_dtype: torch.dtype = torch.bfloat16):
        return cls(
            weight_mx=linear.weight,
            bias=linear.bias,
            out_features=linear.out_features,
            in_features=linear.in_features,
            compute_dtype=compute_dtype,
        )

    def _dequant_weight(self) -> torch.Tensor:
        return _manual_dequantize_mx_tensor(self.weight_mx, self.compute_dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self._dequant_weight()

    @property
    def bias(self) -> torch.Tensor | None:
        if self.bias_buf is None:
            return None
        return self.bias_buf.to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, "
            f"weight=MXTensor(manual_dequant)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hp = self._dequant_weight()
        x = x.to(w_hp.dtype)
        bias = None if self.bias_buf is None else self.bias_buf.to(w_hp.dtype)
        return F.linear(x, w_hp, bias)


class ExportableDequantLinear(nn.Module):
    """
    Export-only wrapper for non-MX quantized linears (e.g. int8/AQT/LAQT).
    This materializes a plain high-precision weight up front so export avoids
    tensor-subclass kernels and avoids `out_dtype` HOPs.
    """

    def __init__(
        self,
        weight_hp: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype

        self.register_buffer("weight_buf", weight_hp.detach().to(compute_dtype), persistent=True)
        if bias is None:
            self.register_buffer("bias_buf", None, persistent=True)
        else:
            self.register_buffer("bias_buf", bias.detach().to(compute_dtype), persistent=True)

    @classmethod
    def from_linear(cls, linear: nn.Linear, compute_dtype: torch.dtype = torch.bfloat16):
        w = linear.weight
        if hasattr(w, "dequantize"):
            w_hp = w.dequantize()
        else:
            w_hp = w
        return cls(
            weight_hp=w_hp,
            bias=linear.bias,
            out_features=linear.out_features,
            in_features=linear.in_features,
            compute_dtype=compute_dtype,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_buf

    @property
    def bias(self) -> torch.Tensor | None:
        return self.bias_buf

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, "
            f"weight=dequantized"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weight_buf.dtype)
        bias = None if self.bias_buf is None else self.bias_buf.to(self.weight_buf.dtype)
        return F.linear(x, self.weight_buf, bias)


def _iter_named_children_with_parent(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        fqn = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, child, fqn
        yield from _iter_named_children_with_parent(child, fqn)


def rewrite_quantized_linears_for_export_(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> list[ReplaceRecord]:
    replacements: list[ReplaceRecord] = []

    to_replace: list[tuple[nn.Module, str, nn.Module, str]] = []
    for parent, child_name, child, fqn in _iter_named_children_with_parent(model):
        if isinstance(child, nn.Linear):
            w = child.weight
            if is_mx_tensor(w) or is_other_quantized_tensor(w):
                to_replace.append((parent, child_name, child, fqn))

    for parent, child_name, child, fqn in to_replace:
        try:
            if is_mx_tensor(child.weight):
                new_mod = ExportableMXLinear.from_linear(child, compute_dtype=compute_dtype)
                reason = "mx->exportable_manual"
            else:
                new_mod = ExportableDequantLinear.from_linear(child, compute_dtype=compute_dtype)
                reason = "int8/aqt->dequantized_export"
            setattr(parent, child_name, new_mod)
            replacements.append(ReplaceRecord(fqn=fqn, replaced=True, reason=reason))
            if verbose:
                print(f"[exportable_linear] replaced {fqn} ({reason})")
        except Exception as e:
            replacements.append(ReplaceRecord(fqn=fqn, replaced=False, reason=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"[exportable_linear] failed {fqn}: {type(e).__name__}: {e}")

    return replacements


def clone_and_rewrite_quantized_linears_for_export(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> tuple[nn.Module, list[ReplaceRecord]]:
    model_copy = copy.deepcopy(model)
    records = rewrite_quantized_linears_for_export_(model_copy, compute_dtype=compute_dtype, verbose=verbose)
    return model_copy, records