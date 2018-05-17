# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._apply_operation import apply_tile
from ....common._registration import register_converter


def convert_sequence_repeat(scope, operator, container):
    repeat_count = operator.raw_operator.sequenceRepeat.nRepetitions
    apply_tile(scope, operator.input_full_names, operator.output_full_names, container,
               operator_name=operator.full_name, repeats=[repeat_count])


register_converter('sequenceRepeat', convert_sequence_repeat)
