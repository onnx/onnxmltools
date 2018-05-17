# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Flatten
from ...common._registration import register_converter


def convert_keras_flatten(scope, operator, container):
    op_type = 'Flatten'
    attrs = {'name': operator.full_name, 'axis': 0}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter(Flatten, convert_keras_flatten)
