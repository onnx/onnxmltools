# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator, register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers


def calculate_regressor_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = (operator.inputs[0].type.shape[0]
         if len(operator.inputs[0].type.shape) > 0 else None)
    operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('LibSvmSVR', calculate_regressor_output_shapes)
