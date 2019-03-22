# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from onnxutils.onnxtk.common._registration import register_shape_calculator
from onnxutils.onnxtk.common.data_types import FloatTensorType
from onnxutils.onnxtk.common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_padding_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C, H', W']
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.padding
    if len(params.paddingAmounts.borderAmounts) > 0:
        output_shape[2] += params.paddingAmounts.borderAmounts[0].startEdgeSize
        output_shape[2] += params.paddingAmounts.borderAmounts[0].endEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].startEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].endEdgeSize

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('padding', calculate_padding_output_shapes)
