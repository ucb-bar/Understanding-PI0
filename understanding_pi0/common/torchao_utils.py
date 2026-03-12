from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from torchao.quantization import (
    FqnToConfig,
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)

try:
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
except ImportError:
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig

try:
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
except ImportError:
    from torchao.quantization.quantize_.common import KernelPreference


@dataclass
class QuantizeResult:
    fqn: str
    requested: str
    applied: str | None
    ok: bool
    error: str | None = None


def make_mx_config() -> MXDynamicActivationMXWeightConfig:
    # EMULATED is the right default for development on non-Blackwell GPUs.
    kernel_pref = getattr(KernelPreference, "EMULATED", KernelPreference.AUTO)
    return MXDynamicActivationMXWeightConfig(
        block_size=32,
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        kernel_preference=kernel_pref,
    )


def make_int8_config() -> Int8DynamicActivationInt8WeightConfig:
    return Int8DynamicActivationInt8WeightConfig()


def list_linear_fqns(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    out: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out.append((name, mod))
    return out


def apply_config_to_exact_fqn_(
    model: nn.Module,
    fqn: str,
    config,
    quant_device: str | None = None,
) -> None:
    cfg = FqnToConfig(fqn_to_config=OrderedDict([(fqn, config)]))
    quantize_(model, cfg, filter_fn=None, device=quant_device)


def safe_quantize_linears_(
    model: nn.Module,
    plan: OrderedDict[str, str | None],
    quant_device: str | None = None,
    verbose: bool = True,
) -> list[QuantizeResult]:
    mx_cfg = make_mx_config()
    int8_cfg = make_int8_config()

    results: list[QuantizeResult] = []

    for fqn, preferred in plan.items():
        if preferred is None:
            results.append(QuantizeResult(fqn=fqn, requested="skip", applied=None, ok=True))
            continue

        if verbose:
            print(f"[quantize] {fqn} -> preferred={preferred}")

        if preferred == "int8":
            try:
                apply_config_to_exact_fqn_(model, fqn, int8_cfg, quant_device=quant_device)
                results.append(QuantizeResult(fqn=fqn, requested="int8", applied="int8", ok=True))
            except Exception as e:
                results.append(
                    QuantizeResult(
                        fqn=fqn,
                        requested="int8",
                        applied=None,
                        ok=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
            continue

        if preferred != "mx":
            results.append(
                QuantizeResult(
                    fqn=fqn,
                    requested=preferred,
                    applied=None,
                    ok=False,
                    error=f"unknown requested quant type: {preferred}",
                )
            )
            continue

        try:
            apply_config_to_exact_fqn_(model, fqn, mx_cfg, quant_device=quant_device)
            results.append(QuantizeResult(fqn=fqn, requested="mx", applied="mx", ok=True))
        except Exception as mx_err:
            if verbose:
                print(f"  [mx failed] {type(mx_err).__name__}: {mx_err}")
                print("  [fallback] retrying with int8")
            try:
                apply_config_to_exact_fqn_(model, fqn, int8_cfg, quant_device=quant_device)
                results.append(
                    QuantizeResult(
                        fqn=fqn,
                        requested="mx",
                        applied="int8",
                        ok=True,
                        error=f"mx_failed={type(mx_err).__name__}: {mx_err}",
                    )
                )
            except Exception as int8_err:
                results.append(
                    QuantizeResult(
                        fqn=fqn,
                        requested="mx",
                        applied=None,
                        ok=False,
                        error=(
                            f"mx_failed={type(mx_err).__name__}: {mx_err}; "
                            f"int8_failed={type(int8_err).__name__}: {int8_err}"
                        ),
                    )
                )

    return results


def summarize_results(results: Iterable[QuantizeResult]) -> dict[str, int]:
    summary = {"mx": 0, "int8": 0, "skipped": 0, "failed": 0}
    for r in results:
        if not r.ok:
            summary["failed"] += 1
        elif r.applied == "mx":
            summary["mx"] += 1
        elif r.applied == "int8":
            summary["int8"] += 1
        else:
            summary["skipped"] += 1
    return summary