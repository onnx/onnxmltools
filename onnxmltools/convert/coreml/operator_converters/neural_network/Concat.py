# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_concat(scope, operator, container):
    op_type = 'Concat'
    attrs = {'name': operator.full_name}
    if operator.raw_operator.concat.sequenceConcat:
        attrs['axis'] = 0
    else:
        attrs['axis'] = 1

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('concat', convert_concat)
