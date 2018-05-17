# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_add


def convert_add(scope, operator, container):
    if len(operator.input_full_names) == 1:
        scaler_name = scope.get_unique_variable_name(operator.full_name + '_B')
        container.add_initializer(scaler_name, onnx_proto.TensorProto.FLOAT, [1], operator.raw_operator.add.alpha)
        inputs = [operator.inputs[0].full_name, scaler_name]
    else:
        inputs = operator.input_full_names

    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        broadcast = 1
    else:
        broadcast = 0

    apply_add(scope, inputs, operator.output_full_names, container, operator_name=operator.full_name,
              broadcast=broadcast)


register_converter('add', convert_add)
