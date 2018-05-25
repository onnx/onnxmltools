# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_truncated_svd(scope, operator, container):
    # Create alias for the scikit-learn truncated SVD model we are going to convert
    svd = operator.raw_operator
    # Transpose [K, C] matrix to [C, K], where C/K is the input/transformed feature dimension
    transform_matrix = svd.components_.transpose()
    transform_matrix_name = scope.get_unique_variable_name('transform_matrix')
    # Put the transformation into an ONNX tensor
    container.add_initializer(transform_matrix_name, onnx_proto.TensorProto.FLOAT,
                              transform_matrix.shape, transform_matrix.flatten())

    # Create the major operator, a matrix multiplication.
    container.add_node('MatMul', [operator.inputs[0].full_name, transform_matrix_name],
                       operator.outputs[0].full_name, name=operator.full_name)


register_converter('SklearnTruncatedSVD', convert_truncated_svd)

