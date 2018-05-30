# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import collections
from ...common._registration import register_converter


def convert_sklearn_linear_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LinearRegressor'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['coefficients'] = op.coef_.tolist()
    attrs['intercepts'] = op.intercept_ if isinstance(op.intercept_, collections.Iterable) else [op.intercept_]
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnElasticNetRegressor', convert_sklearn_linear_regressor)
register_converter('SklearnLinearRegressor', convert_sklearn_linear_regressor)
register_converter('SklearnLinearSVR', convert_sklearn_linear_regressor)
