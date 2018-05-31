# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import collections
import numbers
from keras.layers import Bidirectional
from ...common._registration import register_shape_calculator


def calculate_keras_bidirectional_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], collections.Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None'
                                              for i in op.output_shape[0])
        if op.merge_mode is None:
            operator.outputs[1].type.shape = list(i if isinstance(i, numbers.Integral) else 'None'
                                                  for i in op.output_shape[1])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None' for i in op.output_shape)


register_shape_calculator(Bidirectional, calculate_keras_bidirectional_output_shapes)
