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


class BiDirectionalLSTMLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'biDirectionalLSTM')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):

        # Shape mapping from CoreML to ONNX:
        #   initial_h: ['None', C] ---> [1, C]
        #   initial_c: ['None', C] ---> [1, C]
        #   initial_h_rev: ['None', C] ---> [1, C]
        #   initial_c_rev: ['None', C] ---> [1, C]
        #   Y_h: ['None', C] ---> [1, C]
        #   Y_h: ['None', C] ---> [1, C]
        #   Y_c: ['None', C] ---> [1, C]
        #   Y_h_rev: ['None', C] ---> [1, C]
        #   Y_c_rev: ['None', C] ---> [1, C]
        for top_level_input in context.top_level_inputs:
            dims = []
            if top_level_input.name in inputs[1:] and top_level_input.name:
                dims = extract_dims(top_level_input)
            else:
                continue
            dims[0] = 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_input.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_input.type.tensor_type.shape.dim[i].dim_value = d

        for top_level_output in context.top_level_outputs:
            dims = []
            if top_level_output.name in outputs[1:] and top_level_output.name:
                dims = extract_dims(top_level_input)
            else:
                continue
            dims[0] = 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_output.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_output.type.tensor_type.shape.dim[i].dim_value = d

        # The LSTM inputs are feature vector (X), initial forward hidden state (h_init), initial backward hidden
        # state (h_init_rev), initial forward cell state (c_init), and initial backward cell stat (c_init_rev)e.
        # Because of the differences between ONNX LSTM and CoreML LSTM variable shapes, the bidirectional LSTM 
        # conversion is not straightforward. See some visualizations below for details.
        #
        # Symbols:
        #
        #  C_in: input feature length
        #  C_out: output feature length
        #  S: sequence length. For example, it can be the number of tokens in a sentence.
        #
        #  X: input features of CoreML LSTM. Its shape is [S, C_in].
        #  h_init: initial forward hidden state in CoreML. Its shape is [1, C_out].
        #  c_init: initial forward cell state in CoreML. Its shape is [1, C_out].
        #  h_init_rev: initial backward hidden state in CoreML. Its shape is [1, C_out].
        #  c_init_rev: initial backward cell state in CoreML. Its shape is [1, C_out].
        #  Y: CoreML LSTM's output. It can be [S, C_out] (if sequence_output is on) or [1, C_out] (if sequence_output is off)
        #  Y_h: CoreML LSTM's last foward hidden state. Its shape is [1, C_out].
        #  Y_c: CoreML LSTM's last forward cell state. Its shape is [1, C_out].
        #  Y_h_rev: CoreML LSTM's last backward hidden state. Its shape is [1, C_out].
        #  Y_c_rev: CoreML LSTM's last backward cell state. Its shape is [1, C_out].
        #
        #  X': input features of ONNX LSTM
        #  h_init': initial (forward and backward) hidden states of ONNX LSTM
        #  c_init': initial (forward and backward) hidden states of ONNX LSTM
        #  Y': ONNX LSTM's output
        #  Y_h': ONNX LSTM's last (forward and backward) hidden states
        #  Y_c': ONNX LSTM's last (forward and backward) cell states
        #
        # Computational graph of CoreML bi-directional LSTM (if sequence_output is on):
        #
        #      X [S, C_in]  h_init [1, C_out]  c_init [1, C_out]  h_init_rev [1, C_out]  c_init_rev [1, C_out]
        #      |               |                    |                |                      |
        #      |               '-----------------.  |                |                      |
        #      |                                 |  |                |   .------------------'
        #      '--------------------------------.|  |                |   |
        #                                       ||  |                |   |
        #                                       vv  v                v   v
        #                                       CoreML Bi-directional LSTM
        #                                       ||  |                |   |
        #      .--------------------------------'|  |                |   '------------------.
        #      |                                 |  |                |                      |
        #      |              .------------------'  |                |                      |
        #      |              |                     |                |                      |
        #      v              v                     v                v                      v
        #      Y [S, 2*C_out] Y_h [1, C_out]     Y_c [1, C_out]    Y_h_rev [1, C_out]     Y_c_rev [1, C_out]
        #
        # Computational graph of CoreML bi-directional LSTM in ONNX (if sequence_output is on):
        #
        #      X [S, C_in]  h_init [1, C_out]  h_init_rev [1, C_out] c_init [1, C_out]   c_init_rev [1, C_out]
        #      |               |                    |                  |                      |
        #      |               |                    |                  |                      |
        #      |               '-------.    .-------'                  '--------.    .--------'
        #      |                       |    |                                   |    |
        #      v                       v    v                                   v    v
        #   Reshape                    Concate                                  Concate
        #      |                          |                                        |
        #      v                          v                                        v
        #      X' [S, 1, C_in]        _h_init_ [2, C_out]                      _c_init_ [2, C_out]
        #      |                          |                                        |
        #      |                          v                                        v
        #      |                       Reshape                                  Reshape
        #      |                          |                                        |
        #      |                          v                                        V
        #      |                       h_init' [2, 1, C_out]                    c_init' [2, 1, Cout]
        #      |                          |                                        |
        #      |                          '-----------------.       .--------------'
        #      '----------------------------------.         |       |
        #                                         |         |       |
        #                                         v         v       v
        #                                       ONNX Bi-directional LSTM
        #                                         |         |       |
        #      .----------------------------------'         |       '--------------------.
        #      |                                            |                            |
        #      v                                            v                            v
        #      Y' [S, 2, 1, C_out]                           Y_h' [2, 1, C_out]           Y_c' [2, 1, C_out]
        #      |                                            |                            |
        #      v                                            v                            v
        #   Reshape                                      Reshape                      Reshape
        #      |                                            |                            |
        #      v                                            v                            v
        #      Y  [S, 2*C_out]                             _Y_h_' [2, C_out]            _Y_c_' [2, C_out]
        #                                                   |                            |
        #                                                   v                            v
        #                                                 Split                        Split
        #                                                 |   |                        |   |
        #                     .---------------------------'   |      .-----------------'   |
        #                     |                     .---------'      |                     |
        #                     |                     |                |                     |
        #                     v                     v                v                     v
        #                    Y_h [1, C_out]     Y_h_rev [1, C_out]  Y_c [1, C_out]       Y_c_rev [1, C_out]
        #
        # Computational graph of CoreML bi-directional LSTM (if sequence_output is off):
        #
        #      X [S, C_in]  h_init [1, C_out]  c_init [1, C_out]  h_init_rev [1, C_out]  c_init_rev [1, C_out]
        #      |               |                    |                |                      |
        #      |               '-----------------.  |                |                      |
        #      |                                 |  |                |   .------------------'
        #      '--------------------------------.|  |                |   |
        #                                       ||  |                |   |
        #                                       vv  v                v   v
        #                                       CoreML Bi-directional LSTM
        #                                       ||  |                |   |
        #      .--------------------------------'|  |                |   '------------------.
        #      |                                 |  |                |                      |
        #      |              .------------------'  |                |                      |
        #      |              |                     |                |                      |
        #      v              v                     v                v                      v
        #      Y [1, 2*C_out] Y_h [1, C_out]     Y_c [1, C_out]    Y_h_rev [1, C_out]     Y_c_rev [1, C_out]
        #
        # Computational graph of CoreML bi-directional LSTM in ONNX (if sequence_output is off):
        #
        #      X [S, C_in]  h_init [1, C_out]  h_init_rev [1, C_out] c_init [1, C_out]   c_init_rev [1, C_out]
        #      |               |                    |                  |                      |
        #      |               |                    |                  |                      |
        #      |               '-------.    .-------'                  '--------.    .--------'
        #      |                       |    |                                   |    |
        #      v                       v    v                                   v    v
        #   Reshape                    Concate                                  Concate
        #      |                          |                                        |
        #      v                          v                                        v
        #      X' [S, 1, C_in]        _h_init_ [2, C_out]                      _c_init_ [2, C_out]
        #      |                          |                                        |
        #      |                          v                                        v
        #      |                       Reshape                                  Reshape
        #      |                          |                                        |
        #      |                          v                                        v
        #      |                       h_init' [2, 1, C_out]                    c_init' [2, 1, Cout]
        #      |                          |                                        |
        #      |                          '-----------------.       .--------------'
        #      '----------------------------------.         |       |
        #                                         |         |       |
        #                                         v         v       v
        #                                       ONNX Bi-directional LSTM
        #                                         |         |       |
        #      .----------------------------------'         |       '--------------------.
        #      |                                            |                            |
        #      v                                            v                            v
        #      Y' [S, 2, 1, C_in]          .-------------- Y_h' [2, 1, C_out]           Y_c' [2, 1, C_out]
        #   (useless output)               |                |                            |
        #                                  v                v                            v
        #                               Reshape          Reshape                      Reshape
        #                                  |                |                            |
        #                                  |                v                            v
        #   .------------------------------'              _Y_h_' [2, C_out]            _Y_c_' [2, C_out]
        #   |                                               |                            |
        #   |                                               v                            v
        #   |                                             Split                        Split
        #   |                                             |   |                        |   |
        #   |                 .---------------------------'   |      .-----------------'   |
        #   |                 |                     .---------'      |                     |
        #   |                 |                     |                |                     |
        #   v                 v                     v                v                     v
        #   Y  [1, 2*C_out]   Y_h [1, C_out]     Y_h_rev [1, C_out]  Y_c [1, C_out]       Y_c_rev [1, C_out]

        params = cm_node.biDirectionalLSTM
        lstm_params = params.params
        lstm_weights = params.weightParams
        input_size = params.inputVectorSize
        hidden_size = params.outputVectorSize

        builder_list = []
        pre_nb1 = NodeBuilder(context, 'Reshape')
        builder_list.append(pre_nb1)
        pre_nb1.add_attribute('shape', [-1, 1, input_size])
        pre_nb1.add_input(inputs[0])
        X_name = context.get_unique_name('X')
        pre_nb1.add_output(X_name)

        nb = NodeBuilder(context, 'LSTM')
        builder_list.append(nb)
        nb.add_input(X_name)

        W = np.concatenate([lstm_weights[0].inputGateWeightMatrix.floatValue,
                            lstm_weights[0].outputGateWeightMatrix.floatValue,
                            lstm_weights[0].forgetGateWeightMatrix.floatValue,
                            lstm_weights[0].blockInputWeightMatrix.floatValue])
        WB = np.concatenate([lstm_weights[1].inputGateWeightMatrix.floatValue,
                             lstm_weights[1].outputGateWeightMatrix.floatValue,
                             lstm_weights[1].forgetGateWeightMatrix.floatValue,
                             lstm_weights[1].blockInputWeightMatrix.floatValue])
        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT,
                                          [2, 4 * hidden_size, input_size],
                                          np.concatenate([W, WB]))
        nb.add_initializer(tensor_w)

        R = np.concatenate([lstm_weights[0].inputGateRecursionMatrix.floatValue,
                            lstm_weights[0].outputGateRecursionMatrix.floatValue,
                            lstm_weights[0].forgetGateRecursionMatrix.floatValue,
                            lstm_weights[0].blockInputRecursionMatrix.floatValue])
        RB = np.concatenate([lstm_weights[1].inputGateRecursionMatrix.floatValue,
                             lstm_weights[1].outputGateRecursionMatrix.floatValue,
                             lstm_weights[1].forgetGateRecursionMatrix.floatValue,
                             lstm_weights[1].blockInputRecursionMatrix.floatValue])
        tensor_r = model_util.make_tensor('R', onnx_proto.TensorProto.FLOAT,
                                          [2, 4 * hidden_size, hidden_size],
                                          np.concatenate([R, RB]))
        nb.add_initializer(tensor_r)

        b = np.zeros(shape=(2, 8, hidden_size))

        if lstm_params.hasBiasVectors:
            b[0, 0, :] = lstm_weights[0].inputGateBiasVector.floatValue
            b[0, 1, :] = lstm_weights[0].outputGateBiasVector.floatValue
            b[0, 2, :] = lstm_weights[0].forgetGateBiasVector.floatValue
            b[0, 3, :] = lstm_weights[0].blockInputBiasVector.floatValue
            b[1, 0, :] = lstm_weights[1].inputGateBiasVector.floatValue
            b[1, 1, :] = lstm_weights[1].outputGateBiasVector.floatValue
            b[1, 2, :] = lstm_weights[1].forgetGateBiasVector.floatValue
            b[1, 3, :] = lstm_weights[1].blockInputBiasVector.floatValue

        if lstm_params.forgetBias:
            # One may think we should do something like b[0, 2, :] += 1. and b[1, 2, :] += 1.,
            # but it's not correct as CoreML has added 1 into those bias vectors.
            pass

        if lstm_params.hasBiasVectors or lstm_params.forgetBias:
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT, [2, 8 * hidden_size], b.flatten())
            nb.add_initializer(tensor_b)
        else:
            nb.add_empty_input()

        # sequence_lens
        nb.add_empty_input()

        # initial_h
        if len(inputs) > 1:
            pre_nb2 = NodeBuilder(context, 'Concat')
            builder_list.append(pre_nb2)
            pre_nb2.add_attribute('axis', 0)
            zero_initializer = model_util.make_tensor('h_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb2.add_initializer(zero_initializer, inputs[1])
            zero_initializer = model_util.make_tensor('h_init_rev', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb2.add_initializer(zero_initializer, inputs[3])
            h_init_name = context.get_unique_name('h_init')
            pre_nb2.add_output(h_init_name)

            pre_nb3 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb3)
            pre_nb3.add_attribute('shape', [2, 1, hidden_size])
            h_init_reshaped_name = context.get_unique_name('h_init_reshaped')
            pre_nb3.add_input(h_init_name)
            pre_nb3.add_output(h_init_reshaped_name)

            nb.add_input(h_init_reshaped_name)
        else:
            nb.add_empty_input()

        # initial_c
        if len(inputs) > 2:
            pre_nb4 = NodeBuilder(context, 'Concat')
            builder_list.append(pre_nb4)
            pre_nb4.add_attribute('axis', 0)
            zero_initializer = model_util.make_tensor('c_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb4.add_initializer(zero_initializer, inputs[2])
            zero_initializer = model_util.make_tensor('c_init_rev', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb4.add_initializer(zero_initializer, inputs[4])
            c_init_name = context.get_unique_name('c_init')
            pre_nb4.add_output(c_init_name)

            pre_nb5 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb5)
            pre_nb5.add_attribute('shape', [2, 1, hidden_size])
            pre_nb5.add_input(c_init_name)
            c_init_reshaped_name = context.get_unique_name('c_init_reshaped')
            pre_nb5.add_output(c_init_reshaped_name)

            nb.add_input(c_init_reshaped_name)
        else:
            nb.add_empty_input()

        # peephold vector
        if lstm_params.hasPeepholeVectors:
            p = np.concatenate([lstm_weights[0].inputGatePeepholeVector.floatValue,
                                lstm_weights[0].outputGatePeepholeVector.floatValue,
                                lstm_weights[0].forgetGatePeepholeVector.floatValue])
            pB = np.concatenate([lstm_weights[1].inputGatePeepholeVector.floatValue,
                                 lstm_weights[1].outputGatePeepholeVector.floatValue,
                                 lstm_weights[1].forgetGatePeepholeVector.floatValue])
            tensor_p = model_util.make_tensor('p', onnx_proto.TensorProto.FLOAT,
                                              [2, 3 * hidden_size], np.concatenate([p, pB]))
            nb.add_initializer(tensor_p)
        else:
            nb.add_empty_input()

        activation_types = []
        alphas = []
        betas = []

        for activation in params.activationsForwardLSTM:
            activation_type, alpha, beta = extract_activation_info(activation)
            activation_types.append(activation_type.encode('ascii'))
            if alpha is not None:
                alphas.append(alpha)
            if beta is not None:
                betas.append(beta)

        for activation in params.activationsBackwardLSTM:
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
        nb.add_attribute('direction', 'bidirectional')
        nb.add_attribute('output_sequence', lstm_params.sequenceOutput)
        nb.add_attribute('hidden_size', hidden_size)
        nb.add_attribute('clip', lstm_params.cellClipThreshold if lstm_params.cellClipThreshold != 0 else 50.)
        nb.add_attribute('input_forget', lstm_params.coupledInputAndForgetGate)

        if lstm_params.sequenceOutput:
            Y_name = context.get_unique_name('Y')
            nb.add_output(Y_name)
            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [-1, 2 * hidden_size])
            post_nb1.add_input(Y_name)
            post_nb1.add_output(outputs[0])

            if len(outputs) > 1:
                Y_h_name = context.get_unique_name('Y_h')
                nb.add_output(Y_h_name)
                post_nb2 = NodeBuilder(context, 'Reshape')
                builder_list.append(post_nb2)
                post_nb2.add_attribute('shape', [2, hidden_size])
                post_nb2.add_input(Y_h_name)
                Y_h_reshaped_name = context.get_unique_name('Y_h_reshaped')
                post_nb2.add_output(Y_h_reshaped_name)

                post_nb3 = NodeBuilder(context, 'Split')
                builder_list.append(post_nb3)
                post_nb3.add_attribute('split', [1, 1])
                post_nb3.add_attribute('axis', 0)
                post_nb3.add_input(Y_h_reshaped_name)
                post_nb3.add_output(outputs[1])
                post_nb3.add_output(outputs[3])
        else:
            # Here we ingore ONNX RNN's first output because it's useless.
            nb.add_output(context.get_unique_name('Dummy'))

            # Handle the second output of ONNX LSTM. It will become the first and the second outputs of 
            # CoreML's LSTM.
            Y_h_name = context.get_unique_name('Y_h')
            nb.add_output(Y_h_name)

            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [1, 2 * hidden_size])
            post_nb1.add_input(Y_h_name)
            post_nb1.add_output(outputs[0])

            if len(outputs) > 1:
                post_nb2 = NodeBuilder(context, 'Reshape')
                builder_list.append(post_nb2)
                post_nb2.add_attribute('shape', [2, hidden_size])
                post_nb2.add_input(Y_h_name)
                Y_h_reshaped_name = context.get_unique_name('Y_h_reshaped')
                post_nb2.add_output(Y_h_reshaped_name)

                post_nb3 = NodeBuilder(context, 'Split')
                builder_list.append(post_nb3)
                post_nb3.add_attribute('split', [1, 1])
                post_nb3.add_attribute('axis', 0)
                post_nb3.add_input(Y_h_reshaped_name)
                post_nb3.add_output(outputs[1])
                post_nb3.add_output(outputs[3])

        if len(outputs) > 2:
            Y_c_name = context.get_unique_name('Y_c')
            nb.add_output(Y_c_name)
            post_nb4 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb4)
            post_nb4.add_attribute('shape', [2, hidden_size])
            post_nb4.add_input(Y_c_name)
            Y_c_reshaped_name = context.get_unique_name('Y_c_reshaped')
            post_nb4.add_output(Y_c_reshaped_name)

            post_nb5 = NodeBuilder(context, 'Split')
            builder_list.append(post_nb5)
            post_nb5.add_attribute('split', [1, 1])
            post_nb5.add_attribute('axis', 0)
            post_nb5.add_input(Y_c_reshaped_name)
            post_nb5.add_output(outputs[2])
            post_nb5.add_output(outputs[4])

        return [builder.make_node() for builder in builder_list]


registration.register_nn_converter('biDirectionalLSTM', BiDirectionalLSTMLayerConverter)
