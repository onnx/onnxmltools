# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import TensorType, FloatTensorType
from ....common.utils import check_input_and_output_numbers


def calculate_load_constant_nd_output_shapes(operator):
    check_input_and_output_numbers(
        operator, input_count_range=None, output_count_range=1
    )

    output = operator.outputs[0]

    # CoreML's constant is always 3-D tensor, so we assume its shape is [C, H, W].
    const_shape = operator.raw_operator.loadConstantND.shape
    # We convert [C, H, W] to [1, C, H, W] because our parsing code use [N, C, H, W]
    const_shape = [1] + [int(d) for d in const_shape]
    if output.type is None:
        # Use default type
        output.type = FloatTensorType(const_shape, doc_string=output.type.doc_string)
    else:
        if not isinstance(output.type, TensorType):
            raise RuntimeError("Type conflict detected. Output must be a tensor.")
        # If output type exists, we just modify its shape.
        output.type.shape = const_shape


register_shape_calculator("loadConstantND", calculate_load_constant_nd_output_shapes)
