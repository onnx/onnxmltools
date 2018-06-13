# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from .....proto import onnx_proto
from ....common._registration import register_converter

def convert_inner_product(scope, operator, container):
#TODO: deal with input with shape [N,C,1,1]
    params = operator.raw_operator.innerProduct
    op_type = 'Gemm'
    attrs = {'name': operator.full_name}

    name_a = operator.inputs[0].full_name

    name_b = scope.get_unique_variable_name(operator.full_name + '_B')
    shape_b = [params.outputChannels, params.inputChannels]
    container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.weights.floatValue)

    name_c = scope.get_unique_variable_name(operator.full_name + '_C')
    shape_c = [params.outputChannels]
    if params.hasBias:
        container.add_initializer(name_c, onnx_proto.TensorProto.FLOAT, shape_c, params.bias.floatValue)
    else:
        container.add_initializer(name_c, onnx_proto.TensorProto.FLOAT, shape_c, [0.] * shape_b[0])
    attrs['alpha'] = 1.0
    attrs['beta'] = 1.0
    attrs['transA'] = 0
    attrs['transB'] = 1

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        op_version = 1
    elif container.targeted_onnx_version < StrictVersion('1.2'):
        op_version = 6
    else:
        op_version = 7

    container.add_node(op_type, [name_a, name_b, name_c], operator.outputs[0].full_name, op_version=op_version, **attrs)

register_converter('innerProduct', convert_inner_product)
