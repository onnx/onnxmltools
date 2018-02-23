#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import model_util
from ...common import registration


class EmbeddingLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'embedding')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError(
                'Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        # Type changes of embedding operator's input. They should be integer tensors rather
        # than float tensors by default.
        for top_level_input in context.top_level_inputs:
            onnx_name = context.get_onnx_name(top_level_input.name)
            if onnx_name == inputs[0]:
                top_level_input.type.tensor_type.elem_type = onnx_proto.TensorProto.INT64

        params = cm_node.embedding

        # Reshape the indexes we want to embed to 1-D tensor. Otherwise, Gather's output may get a wrong shape.
        reshaped_input_name = context.get_unique_name(inputs[0]+'_reshaped')
        nb1 = NodeBuilder(context, 'Reshape')
        nb1.add_input(inputs[0])
        nb1.add_output(reshaped_input_name)
        nb1.add_attribute('shape', [-1])

        # Use Gather to extract representations for each index
        nb2 = NodeBuilder(context, 'Gather')
        # Load the embedding matrix. Its shape is outputChannels-by-inputDim.
        weights = np.array(params.weights.floatValue).reshape(params.outputChannels, params.inputDim)
        # Transpose the embedding matrix for applying Gather operator
        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT, [params.inputDim, params.outputChannels],
                                          weights.transpose().flatten().tolist())
        nb2.add_initializer(tensor_w)
        nb2.add_input(reshaped_input_name)

        # To support the bias term in an embedding (if exists), we need to create one extra node
        if params.hasBias:
            # Put the embedded result onto a temporal tensor
            nb2.add_output(nb2.name)
            # Create an addition operator to add bias (shape: [C]) into the temporal tensor (shape: [N, C])
            nb3 = NodeBuilder(context, 'Add')
            nb3.extend_inputs(nb2.output_names)
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT, [
                                              params.outputChannels], params.bias.floatValue)
            nb3.add_initializer(tensor_b)
            nb3.add_attribute('axis', 1)
            nb3.add_attribute('broadcast', 1)
            # Output the result produced by the addition node
            nb3.extend_outputs(outputs)

            return [nb1.make_node(), nb2.make_node(), nb3.make_node()]
        else:
            # This case has no bias, so we just output the result produced by the embedding node.
            nb2.extend_outputs(outputs)

            return [nb1.make_node(), nb2.make_node()]


registration.register_nn_converter('embedding', EmbeddingLayerConverter)
