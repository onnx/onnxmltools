# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_reshape(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReshapeLayerParams as Params

    params = operator.raw_operator.reshape

    if params.mode == Params.CHANNEL_LAST:
        op_type = 'Transpose'
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transpose')
        attrs = {'name': scope.get_unique_operator_name(op_type), 'perm': [0, 2, 3, 1]}
        container.add_node(op_type, [operator.inputs[0].full_name], [intra_variable_name], **attrs)
    else:
        intra_variable_name = operator.inputs[0].full_name

    N = operator.inputs[0].type.shape[0]
    if N == 'None':
        N = -1
    output_shape = [N] + [int(d) for d in params.targetShape]
    desired_shape_name = scope.get_unique_variable_name('shape_tensor')
    container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(output_shape)], output_shape)
    container.add_node('Reshape', [intra_variable_name, desired_shape_name], operator.outputs[0].full_name,
                       op_version=5, name=operator.full_name)


register_converter('reshape', convert_reshape)
