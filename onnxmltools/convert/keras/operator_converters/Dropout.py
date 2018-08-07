# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D

from ...common._apply_operation import apply_identity
from ...common._registration import register_converter


def convert_keras_drop(scope, operator, container):
    apply_identity(scope, operator.inputs[0].full_name, operator.outputs[0].full_name, container)

register_converter(Dropout, convert_keras_drop)
register_converter(SpatialDropout1D, convert_keras_drop)
register_converter(SpatialDropout2D, convert_keras_drop)
register_converter(SpatialDropout3D, convert_keras_drop)
