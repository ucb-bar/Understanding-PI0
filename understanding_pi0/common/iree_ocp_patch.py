from __future__ import annotations

import numpy as np
import torch


def _dtype_is(dtype: torch.dtype, name: str) -> bool:
    return hasattr(torch, name) and dtype == getattr(torch, name)


def _is_byte_float_dtype(dtype: torch.dtype) -> bool:
    return any(
        _dtype_is(dtype, name)
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
    )


def _is_packed_byte_transport_dtype(dtype: torch.dtype) -> bool:
    return _dtype_is(dtype, "float4_e2m1fn_x2")


def _to_numpy_transport_array(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array for DenseResourceElementsAttr transport.

    IMPORTANT:
    - We preserve the *logical* MLIR tensor element type outside this function.
    - This function only chooses a NumPy-compatible byte representation for the
      backing buffer when PyTorch -> NumPy does not support the source dtype.
    """
    detached = t.detach().contiguous().cpu()

    if detached.dtype == torch.bfloat16:
        return np.array(detached.view(torch.int16))

    if _is_byte_float_dtype(detached.dtype):
        return np.array(detached.view(torch.uint8))

    if _is_packed_byte_transport_dtype(detached.dtype):
        return np.array(detached.view(torch.uint8))

    return np.array(detached)


def _patch_fx_importer(verbose: bool = True) -> None:
    from iree.compiler.extras import fx_importer

    try:
        from iree.compiler.ir import IntegerType
    except Exception:
        IntegerType = None

    patched: list[str] = []

    ocp_fp8 = getattr(torch, "float8_e8m0fnu", None)
    ocp_fp4 = getattr(torch, "float4_e2m1fn_x2", None)

    # Keep native upstream mappings for standard float8_e4m3fn/e5m2 dtypes.
    # Only patch exotic OCP dtypes here.

    if hasattr(fx_importer, "TORCH_DTYPE_TO_MLIR_TYPE_ASM"):
        if ocp_fp8 is not None:
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[ocp_fp8] = "ui8"
            patched.append("fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[float8_e8m0fnu]='ui8'")
        if ocp_fp4 is not None:
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[ocp_fp4] = "ui8"
            patched.append("fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[float4_e2m1fn_x2]='ui8'")

    if IntegerType is not None and hasattr(fx_importer, "TORCH_DTYPE_TO_MLIR_TYPE"):
        if ocp_fp8 is not None:
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE[ocp_fp8] = lambda: IntegerType.get_unsigned(8)
            patched.append("fx_importer.TORCH_DTYPE_TO_MLIR_TYPE[float8_e8m0fnu]=ui8")
        if ocp_fp4 is not None:
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE[ocp_fp4] = lambda: IntegerType.get_unsigned(8)
            patched.append("fx_importer.TORCH_DTYPE_TO_MLIR_TYPE[float4_e2m1fn_x2]=ui8")

    for dict_name in ("TORCH_DTYPE_TO_NPY_TYPE", "TORCH_DTYPE_TO_NUMPY_DTYPE"):
        if hasattr(fx_importer, dict_name):
            d = getattr(fx_importer, dict_name)
            # Keep bf16 transport fix.
            d[torch.bfloat16] = np.int16
            patched.append(f"fx_importer.{dict_name}[bfloat16]=np.int16")
            # Only exotic OCP fallback here; standard float8 is already supported upstream.
            if ocp_fp8 is not None:
                d[ocp_fp8] = np.uint8
                patched.append(f"fx_importer.{dict_name}[float8_e8m0fnu]=np.uint8")
            if ocp_fp4 is not None:
                d[ocp_fp4] = np.uint8
                patched.append(f"fx_importer.{dict_name}[float4_e2m1fn_x2]=np.uint8")

    if hasattr(fx_importer, "TORCH_DTYPE_TO_INT"):
        if ocp_fp8 is not None:
            fx_importer.TORCH_DTYPE_TO_INT[ocp_fp8] = 27
            patched.append("fx_importer.TORCH_DTYPE_TO_INT[float8_e8m0fnu]=27")

    if verbose:
        print("[patch] fx_importer")
        for item in patched:
            print(f"  - {item}")


def _patch_turbine_conversions(verbose: bool = True) -> None:
    import iree.turbine.support.conversions as conversions
    from iree.runtime import HalElementType
    from iree.turbine.support.ir_imports import IntegerType

    patched: list[str] = []

    ocp_fp8 = getattr(torch, "float8_e8m0fnu", None)
    ocp_fp4 = getattr(torch, "float4_e2m1fn_x2", None)

    if ocp_fp8 is not None:
        conversions.TORCH_DTYPE_TO_IREE_TYPE[ocp_fp8] = lambda: IntegerType.get_signless(8)
        conversions.TORCH_DTYPE_TO_IREE_TYPE_ASM[ocp_fp8] = "i8"
        conversions.DTYPE_TO_ELEMENT_TYPE[ocp_fp8] = HalElementType.UINT_8
        conversions.TORCH_DTYPE_TO_NUMPY[ocp_fp8] = np.dtype("u1")
        patched.append("turbine conversions patched for float8_e8m0fnu")

    if ocp_fp4 is not None:
        conversions.TORCH_DTYPE_TO_IREE_TYPE[ocp_fp4] = lambda: IntegerType.get_signless(8)
        conversions.TORCH_DTYPE_TO_IREE_TYPE_ASM[ocp_fp4] = "i8"
        conversions.DTYPE_TO_ELEMENT_TYPE[ocp_fp4] = HalElementType.UINT_8
        conversions.TORCH_DTYPE_TO_NUMPY[ocp_fp4] = np.dtype("u1")
        patched.append("turbine conversions patched for float4_e2m1fn_x2")

    conversions.TORCH_DTYPE_TO_NUMPY[torch.bfloat16] = np.dtype("i2")
    patched.append("turbine conversions patched for bfloat16 transport")

    if verbose:
        print("[patch] turbine.support.conversions")
        for item in patched:
            print(f"  - {item}")
            
def _patch_fx_importer_out_dtype_hop(verbose: bool = True) -> None:
    from iree.compiler.extras import fx_importer

    # Don't patch twice.
    if getattr(fx_importer.GraphNodeImporter, "_understanding_pi0_out_dtype_patch", False):
        if verbose:
            print("[patch] fx_importer out_dtype HOP support already patched")
        return

    def _import_hop_out_dtype(self, loc, node, hop):
        """
        Lower higher_order.out_dtype(op, out_dtype, *args) by importing the wrapped
        op directly and using the HOP node's metadata-derived result types.

        This is a pragmatic importer lowering that removes the HOP wrapper.
        """
        if len(node.args) < 2:
            raise NotImplementedError(
                f"Malformed out_dtype HOP node: expected at least 2 args, got {len(node.args)}"
            )

        wrapped_op = node.args[0]
        requested_out_dtype = node.args[1]
        wrapped_args = tuple(node.args[2:])

        if not isinstance(wrapped_op, fx_importer.TorchOpOverload):
            raise NotImplementedError(
                f"out_dtype currently expects arg0 to be a TorchOpOverload, got {type(wrapped_op)}"
            )

        schema = wrapped_op._schema
        assert isinstance(schema, fx_importer.FunctionSchema)

        mlir_op_name = fx_importer._get_mlir_op_name_for_schema(schema)

        # Important: use the HOP node's metadata to determine result type(s),
        # not the wrapped op's default dtype behavior.
        result_types = self._unpack_node_result_types(node, schema)
        if len(result_types) > 1:
            self._multi_result_nodes.add(node)

        operands = []
        for i, parameter in enumerate(schema.arguments):
            if i < len(wrapped_args):
                operands.append(
                    self._import_argument(loc, wrapped_args[i], parameter.type)
                )
            elif parameter.name in node.kwargs:
                operands.append(
                    self._import_argument(
                        loc, node.kwargs[parameter.name], parameter.type
                    )
                )
            else:
                operands.append(
                    self._import_default_value(
                        loc, parameter.default_value, parameter.type
                    )
                )

        operation = fx_importer._emit_operation(
            mlir_op_name,
            result_types=result_types,
            operands=operands,
            loc=loc,
        )

        for i, value in enumerate(operation.results):
            self.bind_node_value(node, value, i)

    fx_importer.GraphNodeImporter._import_hop_out_dtype = _import_hop_out_dtype
    fx_importer.GraphNodeImporter._understanding_pi0_out_dtype_patch = True

    if verbose:
        print("[patch] fx_importer GraphNodeImporter._import_hop_out_dtype")


def _patch_module_builder_create_tensor_global(verbose: bool = True) -> None:
    import iree.turbine.aot.support.ir_utils as ir_utils

    if getattr(ir_utils.ModuleBuilder.create_tensor_global, "_mx_ocp_patch_applied", False):
        if verbose:
            print("[patch] create_tensor_global already patched")
        return

    def patched_create_tensor_global(
        self,
        symbol_name: str,
        t: torch.Tensor,
        *,
        attrs,
        logical_name: str | None = None,
    ):
        element_type = self.torch_dtype_to_iree_type(t.dtype)
        external, external_scope, external_name = attrs.infer_external_from_tensor(t)
        device = ir_utils.DeviceTensorTrait.get(t)

        with ir_utils.InsertionPoint.at_block_begin(self.body), ir_utils.Location.unknown():
            tensor_type = ir_utils.RankedTensorType.get(list(t.shape), element_type)
            ir_attrs = {
                "sym_name": ir_utils.StringAttr.get(symbol_name),
                "sym_visibility": ir_utils.StringAttr.get("private"),
                "type": ir_utils.TypeAttr.get(tensor_type),
            }

            if attrs.noinline:
                ir_attrs["noinline"] = ir_utils.UnitAttr.get()
            if attrs.mutable:
                ir_attrs["is_mutable"] = ir_utils.UnitAttr.get()

            if device:
                if device.queues is None:
                    ir_attrs["stream.affinity"] = ir_utils.Attribute.parse(
                        f"#hal.device.promise<@__device_{device.ordinal}>"
                    )
                else:
                    queues = ", ".join(device.queues)
                    ir_attrs["stream.affinity"] = ir_utils.Attribute.parse(
                        f"#hal.device.promise<@__device_{device.ordinal}, [{queues}]>"
                    )

            if external:
                external_scope_attr = ir_utils.StringAttr.get(external_scope or "model")
                external_name = (
                    external_name
                    if external_name is not None
                    else attrs.map_name(logical_name if logical_name is not None else symbol_name)
                )
                external_name_attr = ir_utils.StringAttr.get(external_name)
                ir_attrs["initial_value"] = ir_utils.Attribute.parse(
                    f"#flow.parameter.named<{external_scope_attr}::{external_name_attr}> : {tensor_type}"
                )
            elif attrs.uninitialized:
                ir_attrs["initial_value"] = ir_utils.Attribute.parse(
                    f"#util.uninitialized : {tensor_type}"
                )
            else:
                # IMPORTANT: preserve tensor_type above, but materialize a
                # NumPy-compatible transport buffer here.
                array = _to_numpy_transport_array(t)
                contents = memoryview(array)
                blob_name = symbol_name
                elements_attr = ir_utils.DenseResourceElementsAttr.get_from_buffer(
                    contents, blob_name, tensor_type
                )
                ir_attrs["initial_value"] = elements_attr

            global_op = ir_utils.Operation.create("util.global", attributes=ir_attrs)
            self.symbol_table.insert(global_op)
            if self.last_global_op is not None:
                global_op.move_after(self.last_global_op)
            self.last_global_op = global_op
            actual_symbol_name = ir_utils.StringAttr(global_op.attributes["sym_name"]).value
            return actual_symbol_name, global_op, tensor_type

    patched_create_tensor_global._mx_ocp_patch_applied = True
    ir_utils.ModuleBuilder.create_tensor_global = patched_create_tensor_global

    if verbose:
        print("[patch] iree.turbine.aot.support.ir_utils.ModuleBuilder.create_tensor_global")


def apply_all_iree_ocp_patches(verbose: bool = True) -> None:
    _patch_fx_importer(verbose=verbose)
    _patch_fx_importer_out_dtype_hop(verbose=verbose)
    _patch_turbine_conversions(verbose=verbose)
    _patch_module_builder_create_tensor_global(verbose=verbose)