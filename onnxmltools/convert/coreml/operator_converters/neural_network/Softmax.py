# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter


def convert_softmax(scope, operator, container):
    op_type = "Softmax"
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {"name": operator.full_name}
    container.add_node(op_type, inputs, outputs, **attrs)


register_converter("softmax", convert_softmax)
