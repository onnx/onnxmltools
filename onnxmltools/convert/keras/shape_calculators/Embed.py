# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Embedding
from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType


def calculate_keras_embed_output_shapes(operator):
    shape = operator.inputs[0].type.shape
    doc_string = operator.inputs[0].type.doc_string
    operator.inputs[0].type = Int64TensorType(shape, doc_string)


register_shape_calculator(Embedding, calculate_keras_embed_output_shapes)
