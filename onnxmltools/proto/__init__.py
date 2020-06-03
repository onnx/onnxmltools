# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Rather than using ONNX protobuf definition throughout our codebase, we import ONNX protobuf definition here so that
# we can conduct quick fixes by overwriting ONNX functions without changing any lines elsewhere.
from onnx import onnx_pb as onnx_proto  # noqa
from onnx import helper

# Overwrite the make_tensor defined in onnx.helper because of a bug (string tensor get assigned twice)
from onnx import mapping
from onnx.onnx_pb import TensorProto
from onnx.helper import split_complex_to_pairs


def _check_onnx_version():
    import pkg_resources
    min_required_version = pkg_resources.parse_version('1.0.1')
    current_version = pkg_resources.get_distribution('onnx').parsed_version
    assert current_version >= min_required_version, 'ONNXMLTools requires ONNX version 1.0.1 or a newer one'


_check_onnx_version()


def _make_tensor_fixed(name, data_type, dims, vals, raw=False):
    '''
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
    '''
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name

    if (data_type == TensorProto.COMPLEX64 or
            data_type == TensorProto.COMPLEX128):
        vals = split_complex_to_pairs(vals)
    if raw:
        tensor.raw_data = vals
    else:
        field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
            mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[data_type]]
        getattr(tensor, field).extend(vals)

    tensor.dims.extend(dims)
    return tensor


helper.make_tensor = _make_tensor_fixed
