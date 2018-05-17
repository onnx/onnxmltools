# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from .common import permute_tensor
from ...common._registration import register_converter


def convert_keras_permute(scope, operator, container):
    op = operator.raw_operator
    axes = [0] + list(op.dims)
    permute_tensor(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, axes, container)


register_converter(keras.layers.Permute, convert_keras_permute)
