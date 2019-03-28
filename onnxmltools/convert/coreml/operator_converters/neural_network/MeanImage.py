# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


from .....proto import onnx_proto
from ....common._apply_operation import apply_sub
from ....common._registration import register_converter


def convert_preprocessing_mean_image(scope, operator, container):
    mean_tensor_name = scope.get_unique_variable_name(operator.full_name + '_mean')

    # We assume that the first input's shape is [N, C, H, W] so that the mean image's shape, [C, H, W], can
    # be inferred from the first input's shape.
    container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT,
                              operator.inputs[0].type.shape[1:], operator.raw_operator.meanImage)

    # We assume that the first input variable's shape is [N, C, H, W] while the mean image's shape is [C, H, W]. Thus,
    # broadcasting should be enabled starting with axis=1.
    apply_sub(scope, [operator.inputs[0].full_name, mean_tensor_name], operator.output_full_names, container,
              axis=1, broadcast=1)


register_converter('meanImagePreprocessor', convert_preprocessing_mean_image)
