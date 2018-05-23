# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_add
from .Scale import deduce_broadcast_axis_and_shape


def convert_bias(scope, operator, container):
    # Feed the input (which we are going to add a bias onto) into Add operator. Its shape is [C, H, W] in CoreML but
    # [N, C, H, W] in ONNX.
    params = operator.raw_operator.bias

    # Adjust CoreML's bias shape and find a proper axis for broadcasting
    axis, shape = deduce_broadcast_axis_and_shape(operator.targeted_onnx_version, params.shape)

    # No matter what shape it is, we need "broadcast" on because input shape is 4-D while bias is at most 3-D.
    broadcast = 1  # True

    # Create bias vector as an ONNX tensor
    bias_tensor_name = scope.get_unique_variable_name(operator.full_name + '_B')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, shape, params.bias.floatValue)

    apply_add(scope, [operator.inputs[0].full_name, bias_tensor_name], operator.output_full_names, container,
              operator_name=operator.full_name, axis=axis, broadcast=broadcast)


register_converter('bias', convert_bias)
