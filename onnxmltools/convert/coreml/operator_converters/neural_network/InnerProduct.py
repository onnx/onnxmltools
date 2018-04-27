# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_inner_product(scope, operator, container):
    params = operator.raw_operator.innerProduct
    op_type = 'FC'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    name_w = scope.get_unique_variable_name(operator.full_name + '_W')
    shape_w = [params.outputChannels, params.inputChannels]
    inputs.append(name_w)
    container.add_initializer(name_w, onnx_proto.TensorProto.FLOAT, shape_w, params.weights.floatValue)

    name_b = scope.get_unique_variable_name(operator.full_name + '_B')
    shape_b = [params.outputChannels]
    inputs.append(name_b)
    if params.hasBias:
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)
    else:
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, [0.] * shape_b[0])

    attrs['axis'] = 1
    attrs['axis_w'] = 1

    container.add_node(op_type, inputs, outputs, **attrs)


register_converter('innerProduct', convert_inner_product)
