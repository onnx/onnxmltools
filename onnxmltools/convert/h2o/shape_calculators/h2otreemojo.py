# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_regressor_output_shapes
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common.data_types import (FloatTensorType, StringTensorType, Int64TensorType)


def calculate_h2otree_output_shapes(operator):
    params = operator.raw_operator["params"]
    if params["classifier"]:
        calculate_tree_classifier_output_shapes(operator, params)
    else:
        calculate_linear_regressor_output_shapes(operator)


def calculate_tree_classifier_output_shapes(operator, params):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=2)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    N = operator.inputs[0].type.shape[0]
    nclasses = params["nclasses"]
    operator.outputs[0].type = StringTensorType(shape=[N])
    if nclasses > 1:
        operator.outputs[1].type = FloatTensorType([N, nclasses])


register_shape_calculator('H2OTreeMojo', calculate_h2otree_output_shapes)
