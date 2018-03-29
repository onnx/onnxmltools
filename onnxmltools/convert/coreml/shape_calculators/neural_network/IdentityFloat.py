# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_identical_float_tensor_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('This layer %s can only have one input and one output' % operator.type)

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or type(output.type) != FloatTensorType:
        raise RuntimeError('Input must be float tensor')

    doc_string = output.type.doc_string
    output.type.shape = copy.deepcopy(input.type.shape)  # Similar to identity but only accept floats
    output.type.doc_string = doc_string


# Preprocessing layers in CoreML
register_shape_calculator('scalerPreprocessor', calculate_identical_float_tensor_shapes)
register_shape_calculator('meanImagePreprocessor', calculate_identical_float_tensor_shapes)

# Standard neural network layers
register_shape_calculator('activation', calculate_identical_float_tensor_shapes)
register_shape_calculator('bias', calculate_identical_float_tensor_shapes)
register_shape_calculator('l2normalize', calculate_identical_float_tensor_shapes)
register_shape_calculator('lrn', calculate_identical_float_tensor_shapes)
register_shape_calculator('mvn', calculate_identical_float_tensor_shapes)
register_shape_calculator('scale', calculate_identical_float_tensor_shapes)
register_shape_calculator('softmax', calculate_identical_float_tensor_shapes)
register_shape_calculator('unary', calculate_identical_float_tensor_shapes)
