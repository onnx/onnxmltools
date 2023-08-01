# SPDX-License-Identifier: Apache-2.0

import collections

from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_matmul, apply_exp, apply_add
from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers


def convert_aft_survival_regression(scope, operator, container):
    op = operator.raw_operator

    coefficients = op.coefficients.toArray().astype(float)
    coefficients_tensor = scope.get_unique_variable_name('coefficients_tensor')
    container.add_initializer(coefficients_tensor, onnx_proto.TensorProto.FLOAT, [1, len(coefficients)], coefficients)
    intercepts = op.intercept.astype(float) if isinstance(op.intercept, collections.Iterable) else [float(op.intercept)]
    intercepts_tensor = scope.get_unique_variable_name('intercepts_tensor')
    container.add_initializer(intercepts_tensor, onnx_proto.TensorProto.FLOAT, [len(intercepts)], intercepts)

    matmul_result = scope.get_unique_variable_name('matmul_result_tensor')
    apply_matmul(scope, [operator.input_full_names[0], coefficients_tensor], matmul_result, container)
    add_result = scope.get_unique_variable_name('intercept_added_tensor')
    apply_add(scope, [matmul_result, intercepts_tensor], add_result, container)
    apply_exp(scope, add_result, operator.output_full_names, container)


register_converter('pyspark.ml.regression.AFTSurvivalRegressionModel', convert_aft_survival_regression)


def calculate_aft_survival_regression_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('pyspark.ml.regression.AFTSurvivalRegressionModel',
                          calculate_aft_survival_regression_output_shapes)
