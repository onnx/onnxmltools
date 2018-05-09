# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from .Reshape import apply_reshape
from .....proto import onnx_proto
from ....common._apply_operation import apply_add
from ....common._registration import register_converter


def convert_embedding(scope, operator, container):
    params = operator.raw_operator.embedding
    gather_op_name = scope.get_unique_operator_name('Gather')
    gather_attrs = {'name': gather_op_name}

    # Reshape the indexes we want to embed to 1-D tensor. Otherwise, ONNX Gather's output may get wrong shape.
    reshaped_input_name = scope.get_unique_variable_name(gather_op_name + 'input_reshaped')  # 2nd input of Gather
    apply_reshape(scope, operator.inputs[0].full_name, reshaped_input_name, container, desired_shape=[-1])

    # Load the embedding matrix. Its shape is outputChannels-by-inputDim.
    weights = np.array(params.weights.floatValue).reshape(params.outputChannels, params.inputDim)
    weights_name = scope.get_unique_variable_name(gather_op_name + '_W')  # 1st input of Gather
    container.add_initializer(weights_name, onnx_proto.TensorProto.FLOAT,
                              [params.inputDim, params.outputChannels], weights.transpose().flatten().tolist())

    # To support the bias term in an embedding (if exists), we need to create one extra node.
    if params.hasBias:
        # Put the embedded result onto a temporal tensor
        gather_output_name = scope.get_unique_variable_name(gather_op_name + '_output')
        container.add_node('Gather', [weights_name, reshaped_input_name], gather_output_name, **gather_attrs)

        # Load the bias vector into an initializer
        bias_name = scope.get_unique_variable_name(gather_op_name + '_bias')
        container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT,
                                  [params.outputChannels], params.bias.floatValue)
        # Create an addition operator to add bias (shape: [C]) into Gather's tensor (shape: [N, C])
        apply_add(scope, [gather_output_name, bias_name], operator.outputs[0].full_name, container, axis=1, broadcast=1)
    else:
        # This case has no bias, so we just output the result produced by the embedding node.
        container.add_node('Gather', [weights_name, reshaped_input_name], operator.output_full_names, **gather_attrs)


register_converter('embedding', convert_embedding)
