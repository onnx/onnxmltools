# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import numpy as np
from ....common._apply_operation import apply_affine, apply_mul, apply_div, apply_pad
from ....common._registration import register_converter


def calculate_legacy_pad_amount(H_in, pad_h, k_h, s_h):
    '''
    This function calculate padding amount along H-axis. It can be applied to other axes. It should be only used with
    pooling conversion.

    :param H_in: input dimension along H-axis
    :param pad_h: padding amount at H-axis
    :param k_h: kernel's H-axis dimension
    :param s_h: stride along H-axis
    :return: (top_padding_amount, bottom_padding_amount)
    '''
    # Calculate a common variable
    H_temp = H_in + 2 * pad_h - k_h
    # Pooling output shape under CoerML IncludeLastPixel padding mode
    H_include_last_pad_out = math.ceil(H_temp / s_h) + 1
    # Pooling output shape under valid padding mode
    H_valid_pad_out = math.floor(H_temp / s_h) + 1
    # Amount of values padded at top boundary. For max pooling, the padded value should be "-inf."
    # For average pooling, we should pad zeros.
    pad_t = pad_h
    # Amount of values padded at bottom boundary (add extra pixels so that H_include_last_pad_out = floor( (H_adjusted_out - k_h) / stride) + 1)
    if H_include_last_pad_out > H_valid_pad_out:
        pad_b = pad_h + (s_h - H_temp % s_h)
    else:
        pad_b = pad_h
    # Intermediate result with pad_t values at top and pad_b valules at bottom of the original input
    H_adjusted_out = H_in + pad_t + pad_b
    # Adjust padded result if the original pooling wants to cut off the last output pixel.
    if (H_include_last_pad_out - 1) * s_h >= H_in + pad_h:
        if H_adjusted_out % s_h == 0:
            H_adjusted_out -= s_h
        else:
            H_adjusted_out -= H_adjusted_out % s_h
    return (pad_t, H_adjusted_out - pad_t - H_in)


def create_legacy_pad(scope, input_name, output_name, H_in, W_in, k_h, k_w,
                      s_h, s_w, p_h, p_w, padded_value, container):
    '''
    This function adds one Pad operator into its last argument, which is a Container object. By feeding the output of
    the created Pad operator into Pool operator under valid padding mode, we can achieve the same functionality of
    CoreML' pooling under IncludeLastPixel padding mode.

    :param scope:
    :param input_name:
    :param output_name:
    :param H_in: input dimension along H-axis
    :param W_in: input dimension along W-axis
    :param k_h: kernel's H-axis dimension
    :param k_w: kernel's W-axis dimension
    :param s_h: stride along H-axis
    :param s_w: stride along W-axis
    :param p_h: padding amount at the beginning and the end of H-axis
    :param p_w: padding amount at the beginning and the end of W-axis
    :param padded_value: value used to fill padded area
    :param container: Container object
    '''
    # Add a Pad operator to pre-process 4-D tensor
    pad_t, pad_b = calculate_legacy_pad_amount(H_in, p_h, k_h, s_h)
    pad_l, pad_r = calculate_legacy_pad_amount(W_in, p_w, k_w, s_w)

    # CoreML pooling operator pads only their H- and W-axes. Here we assume the shape of the tensor to be padded
    # is [N, C, H, W], so we have 8 padding amounts
    #     pads = [N_begin_index, C_begin_index, H_begin_index, W_begin_index,
    #             N_end_index,   C_end_index,   H_end_index,   W_end_index]
    # Because only H- and W-axes are padded in CoreML, we leave padding amounts of N- and C-axes zeros.
    pads = [0, 0, pad_t, pad_l, 0, 0, pad_b, pad_r]
    apply_pad(scope, input_name, output_name, container, pads=pads, value=padded_value)


# The conversion of pooling has several possible outcomes. Let's first define some symbols and then discuss their
# ONNX computational graphs case-by-case.
#
# Symbols:
#  X: input 4-D tensor. It should have a shape [N, C, H, W] following CoreML's pooling definition.
#  Y: output tensor identical to CorML's pooling. Its shapes depends on the pooling type applied.
#
# Case 1: global pooling
#  X ---> ONNX Global Pooling ---> Y
# In this case, ONNX's pooling implementation should directly match CoreML's implementation, so it's just a naive
# translation.
#
# Case 2: local max/L2 pooling with same or valid padding
#
#  X ---> ONNX Local Max/L2 Pooling ---> Y
#
# In this case, ONNX's pooling implementation should directly match CoreML's implementation, so it's just a naive
# translation.
#
# Case 3: local max/L2 pooling under CoreML's IncludeLastPixel padding
#
#  X ---> Pad --> X' ---> ONNX Local Max/L2 Pooling ---> Y
#
# CoreML's IncludeLastPixel padding mode is not supported in ONNX's pooling. We combine a Pad
# operator and a pooling to simulate CoreML's behavior. In this case, the Pad takes all padding-related
# parameters from CoreML's pooling while ONNX's pooling is working under valid padding mode.
#
# Case 4: local average pooling with same or valid padding. exclude_pad_area is on.
#
#  X ---> ONNX Local Average Pooling ---> Y
#
# Current ONNX pooling operator just follows Caffe2, so the padded area is naturally excluded when calculating the
# numerator and denumerator of the pixel average covered by the kernel. That is, the translation from CoreML to ONNX
# is trivial.
#
# Case 5: local average pooling with same or valid padding. exclude_pad_area is off.
#
#  X ---> ONNX Local Average Pooling ---> Y' ------------> Mul ---> Y
#  |                                                        ^
#  |                                                        |
#  '---> Scaler ---> Z ---> ONNX L1-norm Pooling ---> Z' ---'
#
#  The Scaler has "alpha=0" and its "beta" is a constant. If "beta=1", the output of the L1-norm pooling, Z', is the
#  effective kernel size applied at each pixel when padded area is excluded. Here we use "beta=1/kerenel_size" so
#  that one value in Z' stands for
#         (the kernel size without padded area) / (the kernel size with padded area)
#  at a pixel. The output Y' is computed with exclude_pad_area=on, so the element-wise multiplication of Y' and Z'
#  is Y.
#
# Case 6: local average pooling with IncludeLastPixel padding. exclude_pad_area is on.
#
#  X ---> Pad ---> X' ---> ONNX Local Average Pooling ---> Y' ------> Div ---> Y
#  |                                                                                         ^
#  |                                                                                         |
#  '---> Scaler ---> Z ---> Pad ---> Z' ---> ONNX L1-norm Pooling ---> Z''
#
#  The Scaler has "alpha=0" and its "beta" is a constant. If "beta=1", the output of the L1-norm pooling, Z'', is
#  the effective kernel size applied at each pixel when padded area is excluded (since Pad fills the
#  padded area with zeros so those padded pixels are not counted by the L1-norm pooling). Here we use
#  "beta=1/kerenel_size" so that one value in Z' stands for
#         (the kernel size without padded area) / (the kernel size with padded area)
#  at a pixel. The output Y' is computed as if exclude_pad_area=on, so the element-wise division of Y' and Z'' is Y.
#
# Case 7: local average pooling with IncludeLastPixel padding. exclude_pad_area is off.
#
#  X ---> Pad --> X' ---> ONNX Local Average Pooling ---> Y
#
# Since Pad operators add zeros to X's margin and the local pooling here is working under valid padding, it's
# equivalent to the situation of exclude_pad_area=off.
def convert_pooling(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import PoolingLayerParams as Params
    from coremltools.proto.NeuralNetwork_pb2 import SamePadding

    params = operator.raw_operator.pooling
    # The input of Pool
    inputs = [variable.full_name for variable in operator.inputs]
    # The output of Pool
    outputs = [variable.full_name for variable in operator.outputs]

    # Handle global pooling mode. This case if much simpler than the conversion of local pooling.
    attrs = {'name': operator.full_name}
    if params.globalPooling:
        pooling_table = {params.MAX: 'GlobalMaxPool',
                         Params.AVERAGE: 'GlobalAveragePool',
                         Params.L2: 'GlobalLpPool'}

        if params.type not in pooling_table:
            raise ValueError('Unsupported pooling type: {}'.format(params.type))

        op_type = pooling_table[params.type]
        if params.type == Params.L2:
            attrs['p'] = 2
            container.add_node(op_type, inputs, outputs, op_version=2, **attrs)
        else:
            container.add_node(op_type, inputs, outputs, **attrs)
        return

    # From here to the end of this function, we will handle local pooling mode
    if params.type == Params.MAX:
        op_type = 'MaxPool'
        if container.target_opset < 8:
            op_version = 1
        elif container.target_opset < 10:
            op_version = 8
        else:
            op_version = 10
            attrs['ceil_mode'] = 0
    elif params.type == Params.AVERAGE:
        op_type = 'AveragePool'
        if container.target_opset < 7:
            op_version = 1
        elif container.target_opset < 10:
            op_version = 7
        else:
            op_version = 10
            attrs['ceil_mode'] = 0
    elif params.type == Params.L2:
        op_type = 'LpPool'
        attrs['p'] = 2
        op_version = 2
    else:
        raise ValueError('Unsupported pooling type: {}'.format(params.type))

    # CoreML default v.s. non-default parameters
    kernel_shape = [3, 3] if len(params.kernelSize) <= 0 else params.kernelSize
    strides = [1, 1] if len(params.stride) <= 0 else params.stride
    attrs['kernel_shape'] = kernel_shape
    attrs['strides'] = strides

    # Set up padding attributes
    pads = None
    auto_pad = None
    pad_type = params.WhichOneof('PoolingPaddingType')
    if pad_type == 'valid':
        if len(params.valid.paddingAmounts.borderAmounts) > 0:
            pads = [0, 0, 0, 0]
            pads[0] = params.valid.paddingAmounts.borderAmounts[0].startEdgeSize
            pads[1] = params.valid.paddingAmounts.borderAmounts[1].startEdgeSize
            pads[2] = params.valid.paddingAmounts.borderAmounts[0].endEdgeSize
            pads[3] = params.valid.paddingAmounts.borderAmounts[1].endEdgeSize
            # If padding amounts are all zero, there should be no padding list.
            if all(pad == 0 for pad in pads):
                pads = None
                auto_pad = 'VALID'
        else:
            auto_pad = 'VALID'
    elif pad_type == 'same':
        if params.same.asymmetryMode == SamePadding.BOTTOM_RIGHT_HEAVY:
            # In CoreML, BOTTOM_RIGHT_HEAVY means that the extra pixel (when not dividable by 2) will be added to the
            # end of an axis. This behavior matches ONNX's SAME_UPPER mode.
            # Reference: https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html#samepadding
            #            https://github.com/onnx/onnx/blob/rel-1.2.1/docs/Operators.md#AveragePool
            auto_pad = 'SAME_UPPER'
        elif params.same.asymmetryMode == SamePadding.TOP_LEFT_HEAVY:
            # In CoreML, TOP_LEFT_HEAVY means that the extra pixel (when not dividable by 2) will be added to the
            # beginning of an axis. This behavior matches ONNX's SAME_LOWER mode.
            # Reference: https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html#samepadding
            #            https://github.com/onnx/onnx/blob/rel-1.2.1/docs/Operators.md#AveragePool
            auto_pad = 'SAME_LOWER'
        else:
            raise ValueError('Unknown asymmetric mode: {}'.format(params.same.asymmetryMode))
    elif pad_type == 'includeLastPixel':
        # Here we use a Pad operator to mimic the behavior of this CoreML padding.
        auto_pad = 'VALID'
        H = operator.inputs[0].type.shape[2]
        W = operator.inputs[0].type.shape[3]
        pad_h = params.includeLastPixel.paddingAmounts[0]
        pad_w = params.includeLastPixel.paddingAmounts[1]
        legacy_padded_tensor_name = scope.get_unique_variable_name('legacy_padded_tensor')
        padded_value = 0. if params.type != Params.MAX else 1 + np.finfo(np.float32).min
        # Create a sub-graph of cases 3, 6, 7: X ---> Pad ---> X'
        create_legacy_pad(scope, inputs[0], legacy_padded_tensor_name, H, W, kernel_shape[0], kernel_shape[1],
                          strides[0], strides[1], pad_h, pad_w, padded_value, container)
        # Set the first input name to the output of Pad so that the following Pool operator won't access the
        # original input.
        inputs[0] = legacy_padded_tensor_name
    else:
        raise ValueError('Unsupported padding mode: {}'.format(pad_type))

    if pads is not None:
        attrs['pads'] = pads
    if auto_pad is not None:
        attrs['auto_pad'] = auto_pad

    if (params.type == Params.AVERAGE and params.avgPoolExcludePadding and pad_type == 'includeLastPixel') or \
            (params.type == Params.AVERAGE and not params.avgPoolExcludePadding and pad_type != 'includeLastPixel'):
        # Case 5 & 6. See comment above.

        # X --> Affine --> Z
        X_name = operator.inputs[0].full_name
        Y_name = operator.outputs[0].full_name
        Z_name = scope.get_unique_variable_name('Z')
        apply_affine(scope, X_name, Z_name, container, alpha=0., beta=1. / (kernel_shape[0] * kernel_shape[1]))

        Z_prime_name = scope.get_unique_variable_name('Z_prime')
        Y_prime_name = scope.get_unique_variable_name('Y_prime')

        if pad_type != 'includeLastPixel':
            # Create the major Pool operator.
            # Associated sub-graph of case 5: X ---> Pool ---> Y'
            container.add_node(op_type, inputs, Y_prime_name, **attrs)

            # Create operators to calculate correction coefficients
            # Associated sub-graph of case 5: Z ---> L1Pool ---> Z'
            lp_pool_attrs = {'name': scope.get_unique_operator_name('LpPool'), 'kernel_shape': kernel_shape,
                             'strides': strides, 'p': 1}
            if pads is not None:
                lp_pool_attrs['pads'] = pads
            if auto_pad is not None:
                lp_pool_attrs['auto_pad'] = auto_pad
            container.add_node('LpPool', Z_name, Z_prime_name, op_version=2, **lp_pool_attrs)

            # Element-wisely apply adjustment coefficients and create the expected CoreML output
            # Associated sub-graph of case 5: Y', Z' ---> Mul ---> Y
            apply_mul(scope, [Y_prime_name, Z_prime_name], Y_name, container, broadcast=0)
        else:
            # Create the major Pool operator
            # Associated sub-graph of case 6: X' ---> Pool ---> Y'
            container.add_node(op_type, inputs, Y_prime_name, **attrs)

            # Create operators to correct Pool's output
            Y_name = operator.outputs[0].full_name
            Z_prime_prime_name = scope.get_unique_variable_name('Z_prime_prime')

            # Pad the constant tensor.
            # Associated sub-graph of case 6: Z ---> Pad ---> Z'
            create_legacy_pad(scope, Z_name, Z_prime_name, operator.inputs[0].type.shape[2],
                              operator.inputs[0].type.shape[3], kernel_shape[0], kernel_shape[1],
                              strides[0], strides[1], params.includeLastPixel.paddingAmounts[0],
                              params.includeLastPixel.paddingAmounts[1], 0., container)

            # Associated sub-graph of case 6: Z' ---> L1Pool ---> Z''
            lp_pool_attrs = {'name': scope.get_unique_operator_name('LpPool'), 'kernel_shape': kernel_shape,
                             'strides': strides, 'p': 1, 'auto_pad': 'VALID'}
            container.add_node('LpPool', Z_prime_name, Z_prime_prime_name, op_version=2, **lp_pool_attrs)

            # Element-wisely apply adjustment coefficients and create the expected CoreML output
            # Associated sub-graph of case 6: Y', Z''  ---> Div ---> Y
            apply_div(scope, [Y_prime_name, Z_prime_prime_name], Y_name, container, broadcast=0)
    else:
        # Create the major Pool operator
        container.add_node(op_type, inputs, outputs, op_version=op_version, **attrs)


register_converter('pooling', convert_pooling)
