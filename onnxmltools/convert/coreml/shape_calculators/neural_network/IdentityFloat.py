# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_identical_float_tensor_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input = operator.inputs[0]
    output = operator.outputs[0]

    doc_string = output.type.doc_string
    output.type.shape = copy.deepcopy(
        input.type.shape
    )  # Similar to identity but only accept floats
    output.type.doc_string = doc_string


# Preprocessing layers in CoreML
register_shape_calculator("scalerPreprocessor", calculate_identical_float_tensor_shapes)
register_shape_calculator(
    "meanImagePreprocessor", calculate_identical_float_tensor_shapes
)

# Standard neural network layers
register_shape_calculator("activation", calculate_identical_float_tensor_shapes)
register_shape_calculator("bias", calculate_identical_float_tensor_shapes)
register_shape_calculator("l2normalize", calculate_identical_float_tensor_shapes)
register_shape_calculator("lrn", calculate_identical_float_tensor_shapes)
register_shape_calculator("mvn", calculate_identical_float_tensor_shapes)
register_shape_calculator("scale", calculate_identical_float_tensor_shapes)
register_shape_calculator("softmax", calculate_identical_float_tensor_shapes)
register_shape_calculator("unary", calculate_identical_float_tensor_shapes)
