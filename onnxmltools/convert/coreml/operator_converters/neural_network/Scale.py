# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from .....proto import onnx_proto
from ....common._apply_operation import apply_add, apply_mul
from ....common._registration import register_converter


def deduce_broadcast_axis_and_shape(targeted_onnx_version, shape):
    # This function is used to calculate the first axis aligned with the scalar and the scalar's ONNX shape for reduce-
    # like operators. Assuming input variable is always a 4-D tensor, we provide a few of examples. If scalar's shape
    # is [1, 2, 3] and input shape is [5, 2, 3, 8], the aligned axis is the [2] (indexed by 1 because indexes are 0-based)
    # in [5, 2, 3, 8], and the desired scalar shape in ONNX is [2, 3] # (the [1] in [1, 2, 3] is redundant and can cause
    # errors in ONNX's boardcasting). If the scaler's shape is [1], no matter what shape the input is, we leave the axis
    # "None" because ONNX operator may automatically handle it. After ONNX-1.2, we adopt Numpy-style broadcasting rule.

    if targeted_onnx_version < StrictVersion('1.2'):
        # Input shape is [N, C, H, W]
        if len(shape) == 1:
            if shape[0] == 1:
                # shape is [] for scalar, we don't specify axis because it's a scalar
                return None, []
            else:
                # shape is [C], alignment starting at C-axis (indexed by 1)
                return 1, shape
        elif len(shape) == 3:
            if shape[0] == 1:
                # shape is [1, H, W], alignment starting at H-axis (indexed by 2)
                return 2, [shape[1], shape[2]]
            else:
                # shape is [C, H, W], alignment starting at C-axis (indexed by 1)
                return 1, shape
    else:
        # Input shape is [N, C, H, W]
        if len(shape) == 1:
            if shape[0] == 1:
                # shape is [1] for scalar, we don't specify axis because it's a scalar
                return None, [1]
            else:
                # shape is [C], alignment starting at C-axis
                return None, [shape[0], 1, 1]
        elif len(shape) == 3:
            if shape[0] == 1:
                # shape is [1, H, W], alignment starting at C-axis
                return None, shape
            else:
                # shape is [C, H, W], alignment starting at C-axis
                return None, shape


def convert_scale(scope, operator, container):
    # In CoreML's ScaleLayer, the input is first scaled by their "scale" attribute and then a "bias" can be added.
    # Symbols:
    #  a: scale attribute in CoreML's ScaleLayer
    #  b: bias attribute in CoreML's ScaleLayer
    #  x: input
    #  y: output
    # The math formulation of ScaleLayer should be
    #  y = a * x + b
    # Therefore, our strategy of composing ScaleLayer is to have one multiplication followed by an addition.
    params = operator.raw_operator.scale
    op1_type = 'Mul'
    scale_axis, scale_shape = deduce_broadcast_axis_and_shape(operator.targeted_onnx_version, params.shapeScale)
    scale_name = scope.get_unique_variable_name(op1_type + '_B')
    container.add_initializer(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, params.scale.floatValue)

    # CoreML is at most 3-D, so we always turn broadcasting on.
    scale_broadcast = 1

    if not params.hasBias:
        # Create a element-wise multiplication and use it to scale the input. The first input is the variable we want
        # to scale while the second input is their multipliers.
        apply_mul(scope, [operator.inputs[0].full_name, scale_name], operator.output_full_names, container,
                  operator_name=operator.full_name, axis=scale_axis, broadcast=scale_broadcast)
    else:
        # Declare a temporal variable to store the scaled input
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_scaled')
        # Create a element-wise multiplication and use it to scale the input and save the result to a temporal variable
        apply_mul(scope, [operator.inputs[0].full_name, scale_name], intra_variable_name, container,
                  operator_name=operator.full_name, axis=scale_axis, broadcast=scale_broadcast)

        # Prepare materials to build an Add operator for adding bias
        bias_axis, bias_shape = deduce_broadcast_axis_and_shape(operator.targeted_onnx_version, params.shapeBias)

        # CoreML is at most 3-D, so we always turn broadcasting on.
        bias_broadcast = 1

        bias_name = scope.get_unique_variable_name(operator.full_name + '_B')
        container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT, bias_shape, params.bias.floatValue)
        # As bias exists, we add the bias into the output of the multiplication and then use the output of addition
        # as the final output of this conversion.
        apply_add(scope, [intra_variable_name, bias_name], operator.output_full_names, container,
                  axis=bias_axis, broadcast=bias_broadcast)


register_converter('scale', convert_scale)
