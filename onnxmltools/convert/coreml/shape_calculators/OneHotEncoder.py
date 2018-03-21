# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .._data_types import FloatTensorType, StringTensorType
from ..registration import register_shape_calculator


def calculate_one_hot_encoder_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('One-hot encoder has only one input and one output')
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
