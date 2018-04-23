# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType, StringTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_lebel_encoder_output_shapes(operator):
    '''
    This function just copy the input shape to the output because label encoder only alters input features' values, not
    their shape.
    '''
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[Int64TensorType, StringTensorType])

    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = Int64TensorType(copy.deepcopy(input_shape))


register_shape_calculator('SklearnLabelEncoder', calculate_sklearn_lebel_encoder_output_shapes)
