# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_add(scope, operator, container):
    op_type = 'Add'
    attrs = {'name': operator.full_name}

    if len(operator.input_full_names) == 1:
        scaler_name = scope.get_unique_variable_name(op_type + '_B')
        container.add_initializer(scaler_name, onnx_proto.TensorProto.FLOAT, [1], operator.raw_operator.add.alpha)
        inputs = [operator.inputs[0].full_name, scaler_name]
    else:
        inputs = operator.input_full_names

    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        attrs['broadcast'] = 1
    else:
        attrs['broadcast'] = 0

    container.add_node(op_type, inputs, operator.output_full_names, **attrs)


register_converter('add', convert_add)
