# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_min(scope, operator, container):
    op_type = 'Min'
    op_name = scope.get_unique_operator_name(op_type)
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)


register_converter('min', convert_min)
