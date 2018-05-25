# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_truncated_svd_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, K]

    Transform feature dimension from C to K
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType], good_output_types=[FloatTensorType])

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Only 2-D tensor(s) can be input(s)')

    N = operator.inputs[0].type.shape[0]
    K = operator.raw_operator.n_components

    operator.outputs[0].type.shape = [N, K]


register_shape_calculator('SklearnTruncatedSVD', calculate_sklearn_truncated_svd_output_shapes)
