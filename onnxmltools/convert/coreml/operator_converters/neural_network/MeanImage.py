# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_preprocessing_mean_image(scope, operator, container):
    mean_tensor_name = scope.get_unique_variable_name(operator.full_name + '_mean')
    container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT,
                              operator.inputs[0].type.shape[1:], operator.raw_operator.meanImage)

    container.add_node('Sub', [operator.inputs[0].full_name, mean_tensor_name], operator.outputs[0].full_name,
                       name=scope.get_unique_operator_name('Sub'), broadcast=1, axis=1)


register_converter('meanImagePreprocessor', convert_preprocessing_mean_image)
