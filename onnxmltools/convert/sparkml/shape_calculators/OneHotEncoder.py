# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxtk.common._registration import register_shape_calculator
from onnxtk.common.data_types import FloatTensorType


def calculate_sparkml_one_hot_encoder_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, C']
        2. [N, 'None'] ---> [N, 'None']
    '''
    op = operator.raw_operator

    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature.
    encoded_slot_sizes = op.categorySizes

    N = operator.inputs[0].type.shape[0]
    # Calculate the output feature length by replacing the count of categorical
    # features with their encoded widths
    if operator.inputs[0].type.shape[1] != 'None':
        C = operator.inputs[0].type.shape[1] - 1 + sum(encoded_slot_sizes)
    else:
        C = 'None'

    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('pyspark.ml.feature.OneHotEncoderModel', calculate_sparkml_one_hot_encoder_output_shapes)
