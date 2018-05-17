# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import collections
import numbers
from keras.layers import LSTM, SimpleRNN, GRU
from ...common._registration import register_shape_calculator


def convert_keras_lstm_output_shapes(operator):
    op = operator.raw_operator
    if isinstance(op.output_shape[0], collections.Iterable):
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None'
                                              for i in op.output_shape[0])
    else:
        operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None' for i in op.output_shape)


register_shape_calculator(LSTM, convert_keras_lstm_output_shapes)
register_shape_calculator(SimpleRNN, convert_keras_lstm_output_shapes)
register_shape_calculator(GRU, convert_keras_lstm_output_shapes)
