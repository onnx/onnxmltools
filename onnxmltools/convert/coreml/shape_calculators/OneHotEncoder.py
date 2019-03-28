# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, StringTensorType
from ...common.utils import check_input_and_output_numbers


def calculate_one_hot_encoder_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, 1] ---> [N, C']

    C' is the total number of categorical values.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    if operator.inputs[0].type.shape[1] != 1 or len(operator.inputs[0].type.shape) > 2:
        raise RuntimeError('Input must be [N, 1]-tensor')

    int_categories = operator.raw_operator.oneHotEncoder.int64Categories.vector
    str_categories = operator.raw_operator.oneHotEncoder.stringCategories.vector

    N = operator.inputs[0].type.shape[0]

    if len(int_categories) > 0:
        operator.outputs[0].type = FloatTensorType([N, len(int_categories)],
                                                   doc_string=operator.outputs[0].type.doc_string)
    elif len(str_categories) > 0 and type(operator.inputs[0].type) == StringTensorType:
        operator.outputs[0].type = FloatTensorType([N, len(str_categories)],
                                                   doc_string=operator.outputs[0].type.doc_string)
    else:
        raise ValueError('Categorical indexes are missing')


register_shape_calculator('oneHotEncoder', calculate_one_hot_encoder_output_shapes)
