# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from keras.layers import Embedding
from ...common._apply_operation import apply_reshape
from ...common._apply_operation import apply_cast
from ...common._registration import register_converter
from ....proto import onnx_proto
from distutils.version import StrictVersion


def convert_keras_embed(scope, operator, container):
    op = operator.raw_operator  # Keras Embedding layer object
    if hasattr(op, 'mask_zero') and op.mask_zero == True:
        raise NotImplementedError("Embedding layer mask_zero attribute cannot be converted")

    # Reshape the indexes we want to embed to 1-D tensor. Otherwise, Gather's output may get wrong shape, which is the
    # same as our CoreML Embedding converter.
    reshaped_input_name = scope.get_unique_variable_name('embedding_reshaped')
    if operator.targeted_onnx_version < StrictVersion('1.2'):
        apply_reshape(scope, operator.inputs[0].full_name, reshaped_input_name, container, desired_shape=[-1])
    else:
        cast0_name = scope.get_unique_variable_name('embedding_cast0')
        cast1_name = scope.get_unique_variable_name('embedding_cast1')
        # workaround for resahpe in ONNX 1.2 not supporting INT64
        apply_cast(scope, operator.inputs[0].full_name, cast0_name, container, to=onnx_proto.TensorProto.DOUBLE)
        apply_reshape(scope, cast0_name, cast1_name, container, desired_shape=[-1])
        apply_cast(scope, cast1_name, reshaped_input_name, container, to=onnx_proto.TensorProto.INT64)

    # Prepare the weight matrix (i.e., the vectors of all input indices) as an initializer so that the following main
    # operator can access it.
    embedding_tensor_name = scope.get_unique_variable_name('W')
    weights = np.array(op.get_weights()[0].T).reshape(op.output_shape[-1], op.input_dim).transpose().flatten().tolist()
    container.add_initializer(embedding_tensor_name, onnx_proto.TensorProto.FLOAT,
                              [op.input_dim, op.output_shape[-1]], weights)

    # Create a Gather operator to extract the latent representation of each index
    op_type = 'Gather'
    attrs = {'name': operator.full_name}
    gather_name = scope.get_unique_variable_name('embedding_gather')
    container.add_node(op_type, [embedding_tensor_name, reshaped_input_name], gather_name, **attrs)
    output_shape = [-1 if dim == 'None' else dim for dim in operator.outputs[0].type.shape]
    apply_reshape(scope, gather_name, operator.output_full_names, container, desired_shape=output_shape)


register_converter(Embedding, convert_keras_embed)
