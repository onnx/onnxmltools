# SPDX-License-Identifier: Apache-2.0

import math
import numbers
from ....common._registration import register_shape_calculator
from ....common.utils import check_input_and_output_numbers


def calculate_convolution_and_pooling_1D_output_shape(
    input_size,
    kernel_size,
    kernel_dilation,
    stride,
    pad_mode,
    pad_head,
    pad_tail,
    output_size=0,
):
    if not isinstance(input_size, numbers.Integral):
        return "None"
    if output_size > 0:
        return int(output_size)  # Must use output_size = 1 for global pooling

    effective_kernel_size = 1 + kernel_dilation * (
        kernel_size - 1
    )  # For pooling, we always have dilation = 1.
    if pad_mode == "valid":
        return int(
            math.floor(
                (input_size + pad_head + pad_tail - effective_kernel_size) / stride
            )
            + 1
        )
    elif pad_mode == "same":
        return int(math.ceil(input_size / stride))
    elif pad_mode == "includeLastPixel":
        if pad_head != pad_tail:
            raise ValueError(
                "Padding amounts at the beginning and the end of an axis must be the same"
            )
        effective_input_size = input_size + pad_head + pad_tail - effective_kernel_size
        out_size = math.ceil(effective_input_size / stride) + 1
        if (out_size - 1) * stride >= input_size + pad_head:
            out_size -= 1
        return out_size
    else:
        raise ValueError("Unknown padding mode: %s" % pad_mode)


def calculate_convolution_transpose_1D_output_shape(
    input_size,
    kernel_size,
    kernel_dilation,
    stride,
    pad_mode,
    pad_head,
    pad_tail,
    output_size=0,
):
    if not isinstance(input_size, numbers.Integral):
        return "None"
    if output_size > 0:
        return output_size

    effective_kernel_size = 1 + kernel_dilation * (kernel_size - 1)
    if pad_mode == "valid":
        return int(
            (input_size - 1) * stride - pad_head - pad_tail + effective_kernel_size
        )
    elif pad_mode == "same":
        return int(input_size * stride)
    else:
        raise ValueError("Unknown padding mode: %s" % pad_mode)


def calculate_convolution_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C, H', W']
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    params = operator.raw_operator.convolution

    input_shape = operator.inputs[0].type.shape
    operator.outputs[0].type.shape = [
        0,
        0,
        0,
        0,
    ]  # Initialize output shape. It will be modified below.
    output_shape = operator.outputs[0].type.shape

    # Adjust N-axis
    output_shape[0] = input_shape[0]

    # Adjust C-axis
    output_shape[1] = params.outputChannels

    # Set up default and non-default parameters
    dilations = [1, 1]
    if len(params.dilationFactor) > 0:
        dilations = [params.dilationFactor[0], params.dilationFactor[1]]
    kernel_shape = [3, 3]
    if len(params.kernelSize) > 0:
        kernel_shape = params.kernelSize
    strides = [1, 1]
    if len(params.stride) > 0:
        strides = params.stride
    specified_output_shape = [0, 0]  # Only used with convolution transpose
    if params.isDeconvolution and len(params.outputShape) > 0:
        specified_output_shape = list(int(i) for i in params.outputShape)
    pad_mode = params.WhichOneof("ConvolutionPaddingType")
    if pad_mode == "valid" and len(params.valid.paddingAmounts.borderAmounts) > 0:
        pad_amounts = params.valid.paddingAmounts.borderAmounts
        pad_heads = [pad_amounts[0].startEdgeSize, pad_amounts[1].startEdgeSize]
        pad_tails = [pad_amounts[0].endEdgeSize, pad_amounts[1].endEdgeSize]
    else:
        # Padding amounts are useless for same
        # padding and valid padding uses [0, 0] by default.
        pad_heads = [0, 0]
        pad_tails = [0, 0]

    # Adjust H- and W-axes
    for i in range(2):
        if params.isDeconvolution:
            output_shape[i + 2] = calculate_convolution_transpose_1D_output_shape(
                input_shape[i + 2],
                kernel_shape[i],
                dilations[i],
                strides[i],
                pad_mode,
                pad_heads[i],
                pad_tails[i],
                specified_output_shape[i],
            )
        else:
            output_shape[i + 2] = calculate_convolution_and_pooling_1D_output_shape(
                input_shape[i + 2],
                kernel_shape[i],
                dilations[i],
                strides[i],
                pad_mode,
                pad_heads[i],
                pad_tails[i],
            )


register_shape_calculator("convolution", calculate_convolution_output_shapes)
