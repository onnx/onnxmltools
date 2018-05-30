# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Concatenate
from ...common._apply_operation import apply_concat
from ...common._registration import register_converter


def convert_keras_conv(scope, operator, container):
    axis = operator.raw_operator.axis
    if axis < 0:
        axis += len(operator.raw_operator.output.shape)
    apply_concat(scope, operator.input_full_names, operator.output_full_names, container,
                 operator_name=operator.full_name, axis=axis)


register_converter(Concatenate, convert_keras_conv)
