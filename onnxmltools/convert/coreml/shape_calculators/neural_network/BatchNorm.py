# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_batch_normalization_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input_shape = operator.inputs[0].type.shape
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    operator.outputs[0].type.shape = copy.deepcopy(operator.inputs[0].type.shape)


register_shape_calculator('batchnorm', calculate_batch_normalization_output_shapes)
