# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Concatenate
from ...common._registration import register_converter


def convert_keras_conv(scope, operator, container):
    op_type = 'Concat'
    attrs = {'name': operator.full_name}
    axis = operator.raw_operator.axis
    if axis < 0:
        axis += len(operator.raw_operator.output.shape)
    attrs['axis'] = axis
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter(Concatenate, convert_keras_conv)
