# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common.data_types import (
    DictionaryType, FloatTensorType, Int64TensorType,
    SequenceType, StringTensorType,
)
from ..common import get_xgb_params


def calculate_xgboost_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=2)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    N = operator.inputs[0].type.shape[0]

    xgb_node = operator.raw_operator
    params = get_xgb_params(xgb_node)
    booster = xgb_node.get_booster()
    atts = booster.attributes()
    ntrees = len(booster.get_dump(with_stats=True, dump_format = 'json'))
    objective = params["objective"]
            
    if objective == "binary:logistic":
        ncl = 2
    else:
        ncl = ntrees // params['n_estimators']
        if objective == "reg:logistic" and ncl == 1:
            ncl = 2
    classes = xgb_node.classes_
    if (np.issubdtype(classes.dtype, np.floating) or
            np.issubdtype(classes.dtype, np.signedinteger)):
        operator.outputs[0].type = Int64TensorType(shape=[N])
    else:
        operator.outputs[0].type = StringTensorType(shape=[N])
    operator.outputs[1].type = operator.outputs[1].type = FloatTensorType([N, ncl])


register_shape_calculator('XGBClassifier', calculate_xgboost_classifier_output_shapes)
