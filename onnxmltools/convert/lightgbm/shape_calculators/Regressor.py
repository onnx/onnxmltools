# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_regressor_output_shapes
from ...common.data_types import FloatTensorType, Int64TensorType


def calculate_lightgbm_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]
        2. [N, C] ---> [N, 1], [N, n_trees]  (when decision_leaf=True)

    This operator produces a scalar prediction for every example in a batch. If the input batch size is N, the output
    shape may be [N, 1]. If decision_leaf is True, a second output of shape [N, n_trees] is produced.
    """
    decision_leaf = getattr(operator, "decision_leaf", False)
    if decision_leaf:
        from ...common.shape_calculator import check_input_and_output_numbers

        check_input_and_output_numbers(
            operator, input_count_range=1, output_count_range=2
        )
        N = operator.inputs[0].type.shape[0]
        op = operator.raw_operator
        if hasattr(op, "n_outputs_"):
            nout = op.n_outputs_
        else:
            nout = 1
        operator.outputs[0].type = FloatTensorType([N, nout])
        n_trees = op.booster_.num_trees()
        operator.outputs[1].type = Int64TensorType(shape=[N, n_trees])
    else:
        calculate_linear_regressor_output_shapes(operator)


register_shape_calculator("LgbmRegressor", calculate_lightgbm_regressor_output_shapes)

