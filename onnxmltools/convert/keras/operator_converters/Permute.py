# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from ...common._apply_operation import apply_transpose
from ...common._registration import register_converter


def convert_keras_permute(scope, operator, container):
    axes = [0] + list(operator.raw_operator.dims)
    apply_transpose(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container, perm=axes)


register_converter(keras.layers.Permute, convert_keras_permute)
