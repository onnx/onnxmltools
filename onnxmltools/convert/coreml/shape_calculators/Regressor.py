# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType, FloatType, Int64Type
from ...common.utils import check_input_and_output_types


def calculate_traditional_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, C']

    The number C' is the length of prediction vector.
    It can be a scalar (C'=1) or a vector (C'>1)
    """
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, Int64TensorType, FloatType, Int64Type],
    )

    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError("Input(s) must be 2-D tensor(s)")

    model_type = operator.raw_operator.WhichOneof("Type")
    if model_type == "glmRegressor":
        glm = operator.raw_operator.glmRegressor
        C = len(glm.weights)
    elif model_type == "treeEnsembleRegressor":
        tree = operator.raw_operator.treeEnsembleRegressor.treeEnsemble
        C = len(tree.basePredictionValue)
    elif model_type == "supportVectorRegressor":
        C = 1
    else:
        raise ValueError(
            "Model should be one of linear model, tree-based model, and support vector machine"
        )

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType(
        [N, C], doc_string=operator.outputs[0].type.doc_string
    )


register_shape_calculator("glmRegressor", calculate_traditional_regressor_output_shapes)
register_shape_calculator(
    "supportVectorRegressor", calculate_traditional_regressor_output_shapes
)
register_shape_calculator(
    "treeEnsembleRegressor", calculate_traditional_regressor_output_shapes
)
