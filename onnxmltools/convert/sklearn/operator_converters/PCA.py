# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_pca(scope, operator, container):
    pca = operator.raw_operator

    transform_matrix = pca.components_.transpose()
    transform_matrix_name = scope.get_unique_variable_name('transform_matrix')

    container.add_initializer(transform_matrix_name, onnx_proto.TensorProto.FLOAT,
                              transform_matrix.shape, transform_matrix.flatten())

    container.add_node('MatMul', [operator.inputs[0].full_name, transform_matrix_name],
                       operator.outputs[0].full_name, name=operator.full_name)


register_converter('SklearnPCA', convert_pca)
