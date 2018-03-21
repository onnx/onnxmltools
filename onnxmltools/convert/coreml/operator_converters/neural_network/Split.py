# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...registration import register_converter


def convert_split(scope, operator, container):
    op_type = 'Split'
    op_name = scope.get_unique_operator_name(op_type)
    # ONNX Split may evenly divide the input along the specified axis if "split" attribute is not specified.
    # Also, CoreML always evenly split the input along C-axis. Consequently, we only need to specify the axis
    # and make sure the number of outputs in ONNX matches that in CoreML.
    attrs = {'name': op_name, 'axis': 1}  # axis=1 means that we split along C-axis
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_version=2, **attrs)


register_converter('split', convert_split)
