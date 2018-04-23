# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ...common._registration import register_converter
from .common import concatenate_variables


def convert_sklearn_imputer(scope, operator, container):
    op_type = 'Imputer'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    op = operator.raw_operator
    attrs['imputed_value_floats'] = op.statistics_
    if isinstance(op.missing_values, str) and op.missing_values == 'NaN':
        attrs['replaced_value_float'] = np.NaN
    elif isinstance(op.missing_values, float):
        attrs['replaced_value_float'] = float(op.missing_values)
    else:
        raise RuntimeError('Unsupported proposed value')

    concatenated_feature = concatenate_variables(scope, operator.inputs, container)
    container.add_node(op_type, concatenated_feature, operator.outputs[0].full_name, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnImputer', convert_sklearn_imputer)
