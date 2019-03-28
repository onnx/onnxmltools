# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_reduce_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, 1, H, W]
        2. [N, C, H, W] ---> [N, C, 1, W]
        3. [N, C, H, W] ---> [N, C, H, 1]
        4. [N, C, H, W] ---> [N, C, 1, 1]
        5. [N, C, H, W] ---> [N, 1, 1, 1]
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    params = operator.raw_operator.reduce

    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params
    # Adjust C-axis
    if params.axis in [Params.CHW, Params.C]:
        output_shape[1] = 1
    # Adjust H-axis
    if params.axis in [Params.CHW, Params.HW, Params.H]:
        output_shape[2] = 1
    # Adjust W-axis
    if params.axis in [Params.CHW, Params.HW, Params.W]:
        output_shape[3] = 1

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('reduce', calculate_reduce_output_shapes)
