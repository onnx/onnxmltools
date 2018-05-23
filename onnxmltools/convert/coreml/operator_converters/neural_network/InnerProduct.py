# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_reshape

def convert_inner_product(scope, operator, container):
    params = operator.raw_operator.innerProduct
    op_type = 'Gemm'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    shape_a = operator.inputs[0].type.shape
    reshaped_tensor_name = scope.get_unique_variable_name('reshape_output')
    apply_reshape(scope, operator.inputs[0].full_name, reshaped_tensor_name, container, desired_shape=[-1, params.inputChannels])
    inputs[0] = reshaped_tensor_name

    name_b = scope.get_unique_variable_name(operator.full_name + '_B')
    shape_b = [params.outputChannels, params.inputChannels]
    inputs.append(name_b)
    container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.weights.floatValue)

    name_c = scope.get_unique_variable_name(operator.full_name + '_C')
    shape_c = [params.outputChannels]
    inputs.append(name_c)
    if params.hasBias:
        container.add_initializer(name_c, onnx_proto.TensorProto.FLOAT, shape_c, params.bias.floatValue)
    else:
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_c, [0.] * shape_b[0])
    attrs['alpha'] = 1.0
    attrs['beta'] = 1.0
    attrs['transB'] = 1

    #workaround for a CNTK GEMM bug
    gemm_output_name = scope.get_unique_variable_name('gemm_output_name')
    apply_reshape(scope, operator.outputs[0].full_name, gemm_output_name, container, desired_shape=[-1, params.outputChannels])

    container.add_node(op_type, inputs, outputs, **attrs)

register_converter('innerProduct', convert_inner_product)
