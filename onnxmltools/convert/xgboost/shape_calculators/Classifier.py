# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ...common._registration import register_shape_calculator
from ...common.shape_calculator import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)
from ...common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from ..common import get_xgb_params


def calculate_xgboost_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=2)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType]
    )
    N = operator.inputs[0].type.shape[0]

    xgb_node = operator.raw_operator
    params = get_xgb_params(xgb_node)
    objective = params["objective"]
    num_class = params.get("num_class", None)

    if objective != "binary:logistic" and num_class is not None:
        ncl = num_class
    else:
        # The booster config only has num_class for multi-class models.
        ncl = 2
    classes = xgb_node.classes_
    if (
        np.issubdtype(classes.dtype, np.floating)
        or np.issubdtype(classes.dtype, np.integer)
        or np.issubdtype(classes.dtype, np.bool_)
    ):
        operator.outputs[0].type = Int64TensorType(shape=[N])
    else:
        operator.outputs[0].type = StringTensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N, ncl])


register_shape_calculator("XGBClassifier", calculate_xgboost_classifier_output_shapes)
register_shape_calculator("XGBRFClassifier", calculate_xgboost_classifier_output_shapes)
