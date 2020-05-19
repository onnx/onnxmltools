# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType
from ...common.utils import check_input_and_output_types


def calculate_non_max_suppression_output_shapes(operator):
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if operator.target_opset < 10:
        raise RuntimeError("nonMaximumSuppression not supported before Opset 10.")

register_shape_calculator('nonMaximumSuppression', calculate_non_max_suppression_output_shapes)
