# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from ...common._registration import register_converter
from .common import concatenate_variables


def convert_sklearn_scaler(scope, operator, container):
    # If there are multiple input variables, we need to combine them as a whole tensor. Integer(s) would be converted
    # to float(s).
    if len(operator.inputs) > 1:
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        feature_name = operator.inputs[0].full_name

    op = operator.raw_operator
    op_type = 'Scaler'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    if isinstance(op, StandardScaler):
        attrs['scale'] = 1.0 / op.scale_
        attrs['offset'] = op.mean_
    elif isinstance(op, RobustScaler):
        C = operator.inputs[0].type.shape[1]
        if op.with_centering:
            attrs['offset'] = op.center_
        else:
            attrs['offset'] = [0.] * C
        if op.with_scaling:
            attrs['scale'] = 1.0 / op.scale_
        else:
            attrs['scale'] = [1.] * C
    elif isinstance(op, MinMaxScaler):
        attrs['scale'] = op.scale_
        attrs['offset'] = -op.min_/(op.scale_ + 1e-8)  # Add 1e-8 to avoid divided by 0
    elif isinstance(op, MaxAbsScaler):
        C = operator.inputs[0].type.shape[1]
        attrs['scale'] = 1.0 / op.scale_
        attrs['offset'] = [0.] * C
    else:
        raise ValueError('Only scikit-learn StandardScaler and RobustScaler are supported but got %s' % type(op))

    container.add_node(op_type, feature_name, operator.outputs[0].full_name, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnRobustScaler', convert_sklearn_scaler)
register_converter('SklearnScaler', convert_sklearn_scaler)
register_converter('SklearnMinMaxScaler', convert_sklearn_scaler)
register_converter('SklearnMaxAbsScaler', convert_sklearn_scaler)
