# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...registration import register_converter


def convert_flatten(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import FlattenLayerParams as Params

    variable_to_be_flattened_name = operator.inputs[0].full_name
    flattened_variable_name = operator.outputs[0].full_name

    if operator.raw_operator.flatten.mode == Params.CHANNEL_LAST:
        op_type = 'Transpose'
        transpose_operator_name = scope.get_unique_operator_name(op_type)
        transpose_attrs = {'name': transpose_operator_name, 'perm': [0, 2, 3, 1]}
        transposed_variable_name = scope.get_unique_variable_name('transposed')

        container.add_node(op_type, [variable_to_be_flattened_name], [transposed_variable_name], **transpose_attrs)
        variable_to_be_flattened_name = transposed_variable_name

    op_type = 'Flatten'
    flatten_attrs = {'name': operator.full_name, 'axis': 1}

    container.add_node(op_type, [variable_to_be_flattened_name], [flattened_variable_name], **flatten_attrs)


register_converter('flatten', convert_flatten)
