# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)
from .Convolution import calculate_convolution_and_pooling_1D_output_shape


def calculate_pooling_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C, H', W']
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input = operator.inputs[0]
    input_shape = operator.inputs[0].type.shape

    if len(input.type.shape) != 4:
        raise RuntimeError("Input must be 4-D float tensor")

    operator.outputs[0].type.shape = [0, 0, 0, 0]
    output_shape = operator.outputs[0].type.shape

    # Adjust N-axis
    output_shape[0] = input_shape[0]

    # Adjust C-axis
    output_shape[1] = input_shape[1]

    params = operator.raw_operator.pooling
    # Set up default and non-default parameters. Notice that
    # they are only set for H- and W-axes.
    # CoreML Pooling doesn't allow dilation, so we use [1, 1]
    # which is equivalent to no dilation.
    dilations = [
        1,
        1,
    ]
    kernel_shape = [3, 3]
    if len(params.kernelSize) > 0:
        kernel_shape = params.kernelSize
    strides = [1, 1]
    if len(params.stride) > 0:
        strides = params.stride
    pad_mode = params.WhichOneof("PoolingPaddingType")
    if pad_mode == "valid" and len(params.valid.paddingAmounts.borderAmounts) > 0:
        pad_amounts = params.valid.paddingAmounts.borderAmounts
        pad_heads = [pad_amounts[0].startEdgeSize, pad_amounts[1].startEdgeSize]
        pad_tails = [pad_amounts[0].endEdgeSize, pad_amounts[1].endEdgeSize]
    elif (
        pad_mode == "includeLastPixel"
        and len(params.includeLastPixel.paddingAmounts) > 0
    ):
        pad_amounts = params.includeLastPixel.paddingAmounts
        pad_heads = [pad_amounts[0], pad_amounts[1]]
        pad_tails = [pad_amounts[0], pad_amounts[1]]
    else:
        # For same padding, padding amounts are not used
        pad_heads = [0, 0]
        pad_tails = [0, 0]

    # Calculate output shape along H- and W-axes
    for i in range(2):
        output_shape[i + 2] = calculate_convolution_and_pooling_1D_output_shape(
            input_shape[i + 2],
            kernel_shape[i],
            dilations[i],
            strides[i],
            pad_mode,
            pad_heads[i],
            pad_tails[i],
            params.globalPooling,
        )


register_shape_calculator("pooling", calculate_pooling_output_shapes)
