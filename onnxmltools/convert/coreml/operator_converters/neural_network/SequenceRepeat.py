# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_tile
from ....common._registration import register_converter


def convert_sequence_repeat(scope, operator, container):
    repeat_count = (
        operator.raw_operator.sequenceRepeat.nRepetitions
    )  # number of copies along N-axis in CoreML

    if len(operator.inputs[0].type.shape) == 4:
        # Number of copies along [N, C, H, W]
        repeats = [int(repeat_count), 1, 1, 1]
    elif len(operator.inputs[0].type.shape) == 2:
        # Number of copies along [N, C]
        repeats = [int(repeat_count), 1]

    apply_tile(
        scope,
        operator.input_full_names[0],
        operator.output_full_names[0],
        container,
        operator_name=operator.full_name,
        repeats=repeats,
    )


register_converter("sequenceRepeat", convert_sequence_repeat)
