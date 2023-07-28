# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_transpose
from ....common._registration import register_converter


def convert_flatten(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import FlattenLayerParams as Params

    variable_to_be_flattened_name = operator.inputs[0].full_name
    flattened_variable_name = operator.outputs[0].full_name

    if operator.raw_operator.flatten.mode == Params.CHANNEL_LAST:
        transposed_variable_name = scope.get_unique_variable_name("transposed")
        apply_transpose(
            scope,
            variable_to_be_flattened_name,
            transposed_variable_name,
            container,
            perm=[0, 2, 3, 1],
        )
        variable_to_be_flattened_name = transposed_variable_name

    op_type = "Flatten"
    flatten_attrs = {"name": operator.full_name, "axis": 1}

    if container.target_opset < 9:
        target_opset = 1
    elif container.target_opset < 11:
        target_opset = 9
    else:
        target_opset = 11

    container.add_node(
        op_type,
        [variable_to_be_flattened_name],
        [flattened_variable_name],
        op_version=target_opset,
        **flatten_attrs
    )


register_converter("flatten", convert_flatten)
