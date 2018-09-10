# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter
import numpy as np


def convert_sklearn_least_squares(scope, operator, container):
    ols = operator.raw_operator
    coef = ols.coef_.T.astype('float32')
    intercept = ols.intercept_.astype('float32')

    coef_name = scope.get_unique_variable_name('coef')
    intercept_name = scope.get_unique_variable_name('intercept')
    matmul_result_name = scope.get_unique_variable_name('matmul_result')
    cast_result_name = scope.get_unique_variable_name('cast_result')

    container.add_initializer(coef_name, onnx_proto.TensorProto.FLOAT,
                              coef.shape, coef)
    container.add_initializer(intercept_name, onnx_proto.TensorProto.FLOAT,
                              [1], [intercept])

    container.add_node('MatMul', [operator.inputs[0].full_name, coef_name],
                       matmul_result_name, name='MatMul')
    container.add_node('Cast', matmul_result_name, 
                        cast_result_name, to=onnx_proto.TensorProto.FLOAT, op_version=7)
    container.add_node('Sum', [cast_result_name, intercept_name],
                        operator.outputs[0].full_name, name='Sum')


register_converter('SklearnLassoLars', convert_sklearn_least_squares)
register_converter('SklearnRidge', convert_sklearn_least_squares)
