# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_div, apply_sub, apply_sqrt
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

    if operator.type == 'SklearnTruncatedSVD': # TruncatedSVD 
        # Create the major operator, a matrix multiplication.
        container.add_node('MatMul', [operator.inputs[0].full_name, transform_matrix_name],
                           operator.outputs[0].full_name, name=operator.full_name)
    else: # PCA
        if svd.mean_ is not None:
            mean_name = scope.get_unique_variable_name('mean')
            sub_result_name = scope.get_unique_variable_name('sub_result')

            container.add_initializer(mean_name, onnx_proto.TensorProto.FLOAT,
                                      svd.mean_.shape, svd.mean_)

            # Subtract mean from input tensor
            apply_sub(scope, [operator.inputs[0].full_name, mean_name],
                      sub_result_name, container, broadcast=1)
        else:
            sub_result_name = operator.inputs[0].full_name
        if svd.whiten:
            explained_variance_name = scope.get_unique_variable_name('explained_variance')
            explained_variance_root_name = scope.get_unique_variable_name('explained_variance_root')
            matmul_result_name = scope.get_unique_variable_name('matmul_result')

            container.add_initializer(explained_variance_name, onnx_proto.TensorProto.FLOAT,
                                      svd.explained_variance_.shape, svd.explained_variance_)

            container.add_node('MatMul', [sub_result_name, transform_matrix_name],
                               matmul_result_name, name=scope.get_unique_operator_name('MatMul'))
            apply_sqrt(scope, explained_variance_name,
                      explained_variance_root_name, container)
            apply_div(scope, [matmul_result_name, explained_variance_root_name],
                      operator.outputs[0].full_name, container, broadcast=1)
        else:
            container.add_node('MatMul', [sub_result_name, transform_matrix_name],
                               operator.outputs[0].full_name, name=scope.get_unique_operator_name('MatMul'))


register_converter('SklearnPCA', convert_truncated_svd)
register_converter('SklearnTruncatedSVD', convert_truncated_svd)
