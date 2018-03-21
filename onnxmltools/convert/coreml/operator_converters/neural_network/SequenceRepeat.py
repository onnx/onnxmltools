# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...registration import register_converter


def convert_sequence_repeat(scope, operator, container):
    op_type = 'Tile'
    attrs = {'name': operator.full_name}
    attrs['tiles'] = operator.raw_operator.sequenceRepeat.nRepetitions
    attrs['axis'] = 0

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('sequenceRepeat', convert_sequence_repeat)
