# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_add


def convert_inner_product(scope, operator, container):
    params = operator.raw_operator.innerProduct

    name_w = scope.get_unique_variable_name(operator.full_name + '_W')
    shape_w = [params.inputChannels, params.outputChannels]
    weights = np.array(params.weights.floatValue).reshape(params.outputChannels, params.inputChannels).transpose()
    container.add_initializer(name_w, onnx_proto.TensorProto.FLOAT, shape_w, weights.flatten())

    # Do Multiply
    matmul_output_name = scope.get_unique_variable_name('matmul_output_name')

    container.add_node('MatMul', [operator.inputs[0].full_name, name_w], matmul_output_name, name=operator.full_name)

    # DO Add
    name_b = scope.get_unique_variable_name(operator.full_name + '_C')
    shape_b = [params.outputChannels]
    if params.hasBias:
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)
    else:
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, [0.] * shape_b[0])

    apply_add(scope, [matmul_output_name, name_b], operator.outputs[0].full_name, container, axis=-1, broadcast=1)


register_converter('innerProduct', convert_inner_product)
