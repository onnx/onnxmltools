# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_permute_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType, StringTensorType],
                                 good_output_types=[FloatTensorType, Int64TensorType, StringTensorType])

    input = operator.inputs[0]
    output = operator.outputs[0]

    axes = [int(i) for i in operator.raw_operator.permute.axis]
    input_shape = copy.deepcopy(input.type.shape)
    output.type.shape = [input_shape[a] for a in axes]


register_shape_calculator('permute', calculate_permute_output_shapes)
