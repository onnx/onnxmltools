# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers.core import Reshape
from ...common._apply_operation import apply_reshape
from ...common._registration import register_converter


def convert_keras_reshape(scope, operator, container):
    op = operator.raw_operator
    apply_reshape(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container,
                  desired_shape=[-1, ] + list(op.target_shape))


register_converter(Reshape, convert_keras_reshape)
