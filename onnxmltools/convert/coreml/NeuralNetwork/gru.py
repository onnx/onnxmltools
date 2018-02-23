#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from ...common import model_util
from .simpleRecurrent import extract_activation_info
from .simpleRecurrent import extract_dims


class GRULayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'gru')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):

        # Shape changes from default to RNN specific:
        #   initial_h: ['None', C] ---> [1, C]
        #   Y_h: ['None', C] ---> [1, C]
        for top_level_input in context.top_level_inputs:
            dims = []
            onnx_name = context.get_onnx_name(top_level_input.name)
            if onnx_name == inputs[1] and top_level_input.name:
                # The shape is [C] in CoreML and loaded as ['None', C]
                dims = extract_dims(top_level_input)
            else:
                continue
            dims[0] = 1  # Replace 'None' with 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_input.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_input.type.tensor_type.shape.dim[i].dim_value = d
        for top_level_output in context.top_level_outputs:
            onnx_name = context.get_onnx_name(top_level_output.name)
            if onnx_name == outputs[1] and top_level_output.name:
                # The shape is [C] in CoreML and loaded as ['None', C]
                dims = extract_dims(top_level_output)
            else:
                continue
            # Replace 'None' with 1
            dims[0] = 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_output.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_output.type.tensor_type.shape.dim[i].dim_value = d

        # The GRU inputs are feature vector, X, and initial hidden state, h_init. Let C_in and C_out denote
        # the input feature length and the output dimension, respectively. Assume that S is the sequence length
        # (i.e., S-axis is the time axis of a sequence). In CorML, the shapes of X and h_init are [S, C_in] and
        # [1, C_out] respectively. In ONNX, the two shapes become [S, N, C_in] (X) and [D, N, C_out]
        # (h_init), where N is the batch size and S is the sequence length (i.e., S-axis is the time axis). To
        # simulate CoreML GRU under ONNX, we need to introduce some extra operators. Note that N=1 and D=1 always
        # hold in ONNX if we are considering GRU from CoreML because there is no batch size in CoreML and CoreML GRU
        # is always uni-directional.
        # 
        # Below we provide a visualization of our conversion for CoreML GRU.
        #
        # Symbols:
        #  
        #  X: input feature vector of CoreML GRU
        #  h_init: initial GRU state of CoreML
        #  Y: CoreML GRU's output. It can be [S, C_out] (if sequence_output is on ) or [1, C_out] (if sequence_output is off)
        #  Y_h: CoreML GRU's last hidden state
        #
        #  X': input features of ONNX GRU
        #  h_init': initial GRU state of ONNX
        #  Y': ONNX GRU's output
        #  Y_h': ONNX GRU's last hidden state
        #
        # Computational graph of CoreML GRU (sequence_output is on):
        #
        # X [S, C_in] ---> CoreML GRU ---> Y [S, C_out]
        #                    ^     |
        #                    |     |
        # h_init [1, C_cou] -'     '---> Y_h [1, C_out]
        #
        # Computational graph we use to repreent CoreML GRU into ONNX (sequence_output is on):
        #
        # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX GRU --> Y' [S, 1, C_out] --> Reshape --> Y [S, C_out]
        #                                                    ^ |
        #                                                    | |
        # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y_h [1, C_out]
        #
        # Computational graph of CoreML GRU (sequence_output is off):
        #
        # X [S, C_in] ---> CoreML GRU ---> Y [1, C_out]
        #                    ^     |
        #                    |     |
        # h_init [1, C_cou] -'     '---> Y_h [1, C_out] Note that in this case, Y=Y_h.
        #
        # Computational graph we use to represent CoreML GRU into ONNX (sequence_output is off):
        #
        # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX GRU --> Y' [S, 1, C_out] (Y' won't be connected with others)
        #                                                    ^ |
        #                                                    | |
        # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
        #                                                                                                  |
        #                                                                                                  v
        #                                                                                               Identity
        #                                                                                                  |
        #                                                                                                  v
        #                                                                                             Y_h [1, C_out]

        params = cm_node.gru
        input_size = params.inputVectorSize
        hidden_size = params.outputVectorSize

        builder_list = []

        pre_nb1 = NodeBuilder(context, 'Reshape')
        builder_list.append(pre_nb1)
        pre_nb1.add_attribute('shape', [-1, 1, input_size])
        pre_nb1.add_input(inputs[0])
        gru_x_name = context.get_unique_name('gru_x')
        pre_nb1.add_output(gru_x_name)

        nb = NodeBuilder(context, 'GRU')
        builder_list.append(nb)
        nb.add_input(gru_x_name)

        matrices_w = np.concatenate([params.updateGateWeightMatrix.floatValue,
                                     params.resetGateWeightMatrix.floatValue,
                                     params.outputGateWeightMatrix.floatValue])
        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT,
                                          [1, 3 * hidden_size, input_size], matrices_w)
        nb.add_initializer(tensor_w)

        matrices_r = np.concatenate([params.updateGateRecursionMatrix.floatValue,
                                     params.resetGateRecursionMatrix.floatValue,
                                     params.outputGateRecursionMatrix.floatValue])
        tensor_r = model_util.make_tensor('R', onnx_proto.TensorProto.FLOAT,
                                          [1, 3 * hidden_size, hidden_size], matrices_r)
        nb.add_initializer(tensor_r)

        if params.hasBiasVectors:
            matrices_b = np.concatenate([params.updateGateBiasVector.floatValue,
                                         params.resetGateBiasVector.floatValue,
                                         params.outputGateBiasVector.floatValue,
                                         np.zeros(3 * hidden_size)])
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT,
                                              [1, 6 * hidden_size], matrices_b)
            nb.add_initializer(tensor_b)
        else:
            nb.add_empty_input()

        # sequence lens
        nb.add_empty_input()

        # inital_h
        if len(inputs) == 2:
            pre_nb2 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb2)
            pre_nb2.add_attribute('shape', [1, 1, hidden_size])
            zero_initializer = model_util.make_tensor('h_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb2.add_initializer(zero_initializer, inputs[1])
            rnn_h_init_name = context.get_unique_name('rnn_h_init')
            pre_nb2.add_output(rnn_h_init_name)
            nb.add_input(rnn_h_init_name)
        else:
            nb.add_empty_input()

        activation_types = []
        alphas = []
        betas = []
        for activation in params.activations:
            activation_type, alpha, beta = extract_activation_info(activation)
            activation_types.append(activation_type.encode('ascii'))
            if alpha is not None:
                alphas.append(alpha)
            if beta is not None:
                betas.append(beta)

        nb.add_attribute('activations', activation_types)
        if alphas:
            nb.add_attribute('activation_alpha', alphas)
        if betas:
            nb.add_attribute('activation_beta', betas)
        nb.add_attribute('direction', 'reverse' if params.reverseInput else 'forward')
        nb.add_attribute('output_sequence', params.sequenceOutput)
        nb.add_attribute('hidden_size', hidden_size)

        if params.sequenceOutput:
            rnn_y_name = context.get_unique_name('Y')
            nb.add_output(rnn_y_name)
            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [-1, hidden_size])
            post_nb1.add_input(rnn_y_name)
            post_nb1.add_output(outputs[0])

            if len(outputs) == 2:
                rnn_h_name = context.get_unique_name('Y_h')
                nb.add_output(rnn_h_name)
                post_nb2 = NodeBuilder(context, 'Reshape')
                builder_list.append(post_nb2)
                post_nb2.add_attribute('shape', [1, hidden_size])
                post_nb2.add_input(rnn_h_name)
                post_nb2.add_output(outputs[1])
        else:
            # Here we ingore ONNX RNN's first output because it's useless.
            nb.add_output(context.get_unique_name('Dummy'))

            # According to Keras and CoreML, the two outputs are always identical, so we just need to
            # compute one of them and produce the other one using identiy operator.
            rnn_h_name = context.get_unique_name('Y_h')
            nb.add_output(rnn_h_name)
            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [1, hidden_size])
            post_nb1.add_input(rnn_h_name)
            post_nb1.add_output(outputs[0])

            # Create CoreML's second output from CoreML's first output
            if len(outputs) == 2:
                post_nb2 = NodeBuilder(context, 'Identity')
                builder_list.append(post_nb2)
                post_nb2.add_input(outputs[0])
                post_nb2.add_output(outputs[1])

        return [builder.make_node() for builder in builder_list]


registration.register_nn_converter('gru', GRULayerConverter)
