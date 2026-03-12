from __future__ import annotations

import importlib.util
from pathlib import Path


PATCH_SENTINEL = "### UNDERSTANDING_PI0_DIFFUSERS_TORCHAO_LOGGER_PATCH ###"


def patch_diffusers_torchao_logger_bug(verbose: bool = True) -> bool:
    """
    Patch diffusers' torchao quantizer so missing uint4 support only emits a warning
    instead of crashing with `logger` / `_get_library_root_logger` NameError.

    The patch is idempotent and rewrites the file in a deterministic way.
    """
    spec = importlib.util.find_spec("diffusers")
    if spec is None or spec.origin is None:
        if verbose:
            print("[hf_compat] diffusers not found; skipping patch")
        return False

    diffusers_root = Path(spec.origin).resolve().parent
    target = diffusers_root / "quantizers" / "torchao" / "torchao_quantizer.py"

    if not target.exists():
        if verbose:
            print(f"[hf_compat] target file not found: {target}")
        return False

    text = target.read_text()

    if PATCH_SENTINEL in text:
        if verbose:
            print(f"[hf_compat] already patched: {target}")
        return False

    patch_block = (
        f"{PATCH_SENTINEL}\n"
        "from ...utils.logging import _get_library_root_logger\n"
        "logger = _get_library_root_logger()\n\n"
    )

    # Remove any broken prior patch fragments that may have been inserted at the top.
    lines = text.splitlines(keepends=True)
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "logger = _get_library_root_logger()":
            continue
        if "_get_library_root_logger" in stripped and "from ...utils.logging import" in stripped:
            continue
        if stripped == PATCH_SENTINEL:
            continue
        cleaned_lines.append(line)

    cleaned = "".join(cleaned_lines)

    # Prepend the correct patch block at the very top.
    new_text = patch_block + cleaned
    target.write_text(new_text)

    if verbose:
        print(f"[hf_compat] patched: {target}")

    return True