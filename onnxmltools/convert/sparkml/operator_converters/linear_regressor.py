# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import collections

from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers


def convert_sparkml_linear_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LinearRegressor'
    attrs = {
        'name': scope.get_unique_operator_name(op_type),
        'coefficients': op.coefficients.astype(float),
        'intercepts': op.intercept.astype(float) if isinstance(op.intercept, collections.Iterable) else [
                 float(op.intercept)]
    }
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.regression.LinearRegressionModel', convert_sparkml_linear_regressor)
register_converter('pyspark.ml.regression.GeneralizedLinearRegressionModel', convert_sparkml_linear_regressor)


def calculate_linear_regressor_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('pyspark.ml.regression.LinearRegressionModel', calculate_linear_regressor_output_shapes)
register_shape_calculator('pyspark.ml.regression.GeneralizedLinearRegressionModel',
                          calculate_linear_regressor_output_shapes)
