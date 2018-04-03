# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from keras.layers import Embedding
from ...common._registration import register_converter
from ....proto import onnx_proto


def convert_keras_embed(scope, operator, container):
    op = operator.raw_operator  # Keras Embedding layer object
    if hasattr(op, 'mask_zero') and op.mask_zero == True:
        raise NotImplementedError("Embedding layer mask_zero attribute cannot be converted")

    # Reshape the indexes we want to embed to 1-D tensor. Otherwise, Gather's output may get wrong shape, which is the
    # same as our CoreML Embedding converter.
    preprocessor_type = 'Reshape'
    preprocessor_attrs = {'name': scope.get_unique_operator_name(preprocessor_type)}
    preprocessor_attrs['shape'] = [-1]
    reshaped_input_name = scope.get_unique_variable_name('embedding_reshaped')
    container.add_node(preprocessor_type, operator.inputs[0].full_name, reshaped_input_name, **preprocessor_attrs)

    # Prepare the weight matrix (i.e., the vectors of all input indices) as an initializer so that the following main
    # operator can access it.
    embedding_tensor_name = scope.get_unique_variable_name('W')
    weights = np.array(op.get_weights()[0].T).reshape(op.output_shape[-1], op.input_dim).transpose().flatten().tolist()
    container.add_initializer(embedding_tensor_name, onnx_proto.TensorProto.FLOAT,
                              [op.input_dim, op.output_shape[-1]], weights)

    # Create a Gather operator to extract the latent representation of each index
    op_type = 'Gather'
    attrs = {'name': operator.full_name}
    container.add_node(op_type, [embedding_tensor_name, reshaped_input_name], operator.output_full_names, **attrs)


register_converter(Embedding, convert_keras_embed)
