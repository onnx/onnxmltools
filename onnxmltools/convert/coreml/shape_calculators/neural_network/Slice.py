# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_slice_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C', H, W]
        2. [N, C, H, W] ---> [N, C, H', W]
        3. [N, C, H, W] ---> [N, C, H, W']
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.slice

    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params

    axis_map = {Params.CHANNEL_AXIS: 1, Params.HEIGHT_AXIS: 2, Params.WIDTH_AXIS: 3}

    if params.startIndex >= 0:
        output_shape[axis_map[Params.CHANNEL_AXIS]] = (
            params.endIndex - params.startIndex
        )
    else:
        output_shape[axis_map[Params.CHANNEL_AXIS]] += (
            1 + params.endIndex - params.startIndex
        )

    operator.outputs[0].type = FloatTensorType(
        output_shape, doc_string=operator.outputs[0].type.doc_string
    )


register_shape_calculator("slice", calculate_slice_output_shapes)
