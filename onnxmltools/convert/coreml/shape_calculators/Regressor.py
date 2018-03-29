# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import FloatTensorType, Int64TensorType, FloatType, Int64Type
from ...common._registration import register_shape_calculator


def calculate_traditional_regressor_output_shapes(operator):
    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be tensor(s)')
    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be 2-D tensor(s)')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'glmRegressor':
        glm = operator.raw_operator.glmRegressor
        C = len(glm.weights)
    elif model_type == 'treeEnsembleRegressor':
        tree = operator.raw_operator.treeEnsembleRegressor.treeEnsemble
        C = len(tree.basePredictionValue)
    elif model_type == 'supportVectorRegressor':
        C = 1
    else:
        raise ValueError('Model should be one of linear model, tree-based model, and support vector machine')

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, C], doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('glmRegressor', calculate_traditional_regressor_output_shapes)
register_shape_calculator('supportVectorRegressor', calculate_traditional_regressor_output_shapes)
register_shape_calculator('treeEnsembleRegressor', calculate_traditional_regressor_output_shapes)
