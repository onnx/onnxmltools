# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...registration import register_converter


def convert_permute(scope, operator, container):
    op_type = 'Transpose'
    attrs = {'name': operator.full_name}
    attrs['perm'] = operator.raw_operator.permute.axis

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('permute', convert_permute)
