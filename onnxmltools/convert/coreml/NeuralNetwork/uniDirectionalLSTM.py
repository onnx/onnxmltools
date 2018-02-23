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


class UniDirectionalLSTMLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'uniDirectionalLSTM')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):

        # Shape changes from default to RNN specific:
        #   initial_h: ['None', C] ---> [1, C]
        #   initial_c: ['None', C] ---> [1, C]
        #   Y_h: ['None', C] ---> [1, C]
        #   Y_h: ['None', C] ---> [1, C]
        #   Y_c: ['None', C] ---> [1, C]
        for top_level_input in context.top_level_inputs:
            onnx_name = context.get_onnx_name(top_level_input.name)
            if onnx_name in inputs[1:] and top_level_input.name:
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
            if onnx_name in outputs[1:] and top_level_output.name:
                # The shape is [C] in CoreML and loaded as ['None', C]
                dims = extract_dims(top_level_output)
            else:
                continue
            dims[0] = 1  # Replace 'None' with 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_output.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_output.type.tensor_type.shape.dim[i].dim_value = d

        # The LSTM inputs are feature vector, X, initial hidden state, h_init, and initial cell state, c_init.
        # In CorML, their shapes respectively are [S, C_in], [1, C_out], and [1, C_out], where C_in is input feature
        # length, # C_out is output dimension, and S is sequence length. Note that S-axis is also known as time axis.
        # In ONNX, those shapes become [S, N, C_in] (X), [D, N, C_out] (h_init), and [D, N, C_out]. To simulate
        # CoreML LSTM under ONNX, we need some extra operators in addition to LSTM itself.
        #
        # Note that N=1 and D=1 are always true in ONNX if we are considering LSTM in CoreML because there is no
        # batch size in CoreML spec and CoreML LSTM is always uni-directional.
        #
        # Below we provide a visualization of our conversion for CoreML LSTM.
        # 
        # Symbols:
        #  
        #  X: input features of CoreML LSTM
        #  h_init: initial LSTM hidden state in CoreML
        #  c_init: initial LSTM cell state in CoreML
        #  Y: CoreML LSTM's output. It can be [S, C_out] (if sequence_output is on) or [1, C_out] (if sequence_output is off)
        #  Y_h: CoreML LSTM's last hidden state
        #  Y_c: CoreML LSTM's last cell state
        #
        #  X': input features of ONNX LSTM
        #  h_init': initial LSTM hidden state of ONNX
        #  c_init': initial LSTM hidden state of ONNX
        #  Y': ONNX LSTM's output
        #  Y_h': ONNX LSTM's last hidden state
        #  Y_c': ONNX LSTM's last cell state
        #
        # Computational graph of CoreML LSTM (if sequence_output is on):
        # 
        #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
        #      |                 |                    |
        #      v                 v                    v
        #      '---------------  |    -----------------
        #                     |  |    |
        #                     v  v    v
        #                    CoreML LSTM
        #                     |  |    |
        #      ---------------'  l    '----------------
        #      |                 |                    |
        #      v                 v                    v
        #      Y [S, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
        #
        # Computational graph of CoreML LSTM in ONNX (if sequence_output is on):
        # 
        #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
        #      |                 |                    |
        #      v                 v                    v
        #   Reshape           Reshape              Reshape
        #      |                 |                    |
        #      v                 v                    v
        #      X'[S, 1, C_in]  h_init'[1, 1, C_out] c_init [1, 1, C_out]
        #      |                 |                    |
        #      '---------------  |    -----------------
        #                     |  |    |
        #                     v  v    v
        #                     ONNX LSTM
        #                     |  |    |
        #       --------------'  vl   '---------------
        #      |                 |                    |
        #      v                 v                    v
        #      Y'[S, 1, C_out]  Y_h' [1, 1, C_out]   Y_c [1, 1, C_out]
        #      |                 |                    |
        #      v                 v                    v
        #   Reshape           Reshape              Reshape
        #      |                 |                    |
        #      v                 v                    v
        #      Y [S, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
        #
        # Computational graph of CoreML LSTM (if sequence_output is off):
        # 
        #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
        #      |                 |                    |
        #      v                 v                    v
        #      '---------------  |    -----------------
        #                     |  |    |
        #                     v  v    v
        #                    CoreML LSTM
        #                     |  |    |
        #      ---------------'  l    '----------------
        #      |                 |                    |
        #      v                 v                    v
        #      Y [1, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
        #           (Note that Y = Y_h)
        #
        # Computational graph of CoreML LSTM in ONNX (if sequence_output is off):
        # 
        #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
        #      |                 |                    |
        #      v                 v                    v
        #   Reshape           Reshape              Reshape
        #      |                 |                    |
        #      v                 v                    v
        #      X'[S, 1, C_in]  h_init'[1, 1, C_out] c_init [1, 1, C_out]
        #      |                 |                    |
        #      '---------------  |    -----------------
        #                     |  |    |
        #                     v  v    v
        #                     ONNX LSTM
        #                     |  |    |
        #       --------------'  vl   '---------------
        #      |                 |                    |
        #      v                 v                    v
        #      Y'[S, 1, C_out]  Y_h' [1, 1, C_out]   Y_c [1, 1, C_out]
        #  (useless output)      |                    |
        #                        v                    v
        #                     Reshape              Reshape
        #                        |                    |
        #                        v                    v
        #                    Y [1, C_out]       Y_c [1, C_out]
        #                        |
        #                        v
        #                     Identity
        #                        |
        #                        v
        #                    Y_h [1, C_out]

        params = cm_node.uniDirectionalLSTM
        lstm_params = params.params
        lstm_weights = params.weightParams
        input_size = params.inputVectorSize
        hidden_size = params.outputVectorSize

        builder_list = []

        lstm_x_name = context.get_unique_name('lstm_x')
        pre_nb1 = NodeBuilder(context, 'Reshape')
        builder_list.append(pre_nb1)
        pre_nb1.add_attribute('shape', [-1, 1, input_size])
        pre_nb1.add_input(inputs[0])
        pre_nb1.add_output(lstm_x_name)

        nb = NodeBuilder(context, 'LSTM')
        builder_list.append(nb)
        nb.add_input(lstm_x_name)

        W = np.concatenate([lstm_weights.inputGateWeightMatrix.floatValue,
                            lstm_weights.outputGateWeightMatrix.floatValue,
                            lstm_weights.forgetGateWeightMatrix.floatValue,
                            lstm_weights.blockInputWeightMatrix.floatValue])
        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT,
                                          [1, 4 * hidden_size, input_size], W)
        nb.add_initializer(tensor_w)

        R = np.concatenate([lstm_weights.inputGateRecursionMatrix.floatValue,
                            lstm_weights.outputGateRecursionMatrix.floatValue,
                            lstm_weights.forgetGateRecursionMatrix.floatValue,
                            lstm_weights.blockInputRecursionMatrix.floatValue])
        tensor_r = model_util.make_tensor('R', onnx_proto.TensorProto.FLOAT,
                                          [1, 4 * hidden_size, hidden_size], R)
        nb.add_initializer(tensor_r)

        b = np.zeros(shape=(8, hidden_size))

        if lstm_params.hasBiasVectors:
            b[0, :] = lstm_weights.inputGateBiasVector.floatValue
            b[1, :] = lstm_weights.outputGateBiasVector.floatValue
            b[2, :] = lstm_weights.forgetGateBiasVector.floatValue
            b[3, :] = lstm_weights.blockInputBiasVector.floatValue

        if lstm_params.forgetBias:
            # One may think we should do something like b[2, :] += 1., but it's wrong as CoreML has
            # added 1 into lstm_weights.forgetGateBiasVector.floatValue.
            pass

        if lstm_params.hasBiasVectors or lstm_params.forgetBias:
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT,
                                              [1, 8 * hidden_size], b.flatten())
            nb.add_initializer(tensor_b)
        else:
            nb.add_empty_input()

        # sequence_lens
        nb.add_empty_input()

        # initial_h
        if len(inputs) > 1:
            lstm_h_init_name = context.get_unique_name('lstm_h_init')
            pre_nb2 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb2)
            pre_nb2.add_attribute('shape', [1, hidden_size])
            zero_initializer = model_util.make_tensor('h_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb2.add_initializer(zero_initializer, inputs[1])
            pre_nb2.add_output(lstm_h_init_name)

            nb.add_input(lstm_h_init_name)
        else:
            nb.add_empty_input()

        # initial_c
        if len(inputs) > 2:
            lstm_c_init_name = context.get_unique_name('lstm_c_init')
            pre_nb3 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb3)
            pre_nb3.add_attribute('shape', [1, hidden_size])
            zero_initializer = model_util.make_tensor('c_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb3.add_initializer(zero_initializer, inputs[2])
            pre_nb3.add_output(lstm_c_init_name)

            nb.add_input(lstm_c_init_name)
        else:
            nb.add_empty_input()

        if lstm_params.hasPeepholeVectors:
            p = np.concatenate([lstm_weights.inputGatePeepholeVector.floatValue,
                                lstm_weights.outputGatePeepholeVector.floatValue,
                                lstm_weights.forgetGatePeepholeVector.floatValue])
            tensor_p = model_util.make_tensor(
                '.p', onnx_proto.TensorProto.FLOAT, [1, 3 * hidden_size], p)
            nb.add_initializer(tensor_p)
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
        nb.add_attribute('output_sequence', lstm_params.sequenceOutput)
        nb.add_attribute('hidden_size', hidden_size)
        nb.add_attribute('clip', lstm_params.cellClipThreshold if lstm_params.cellClipThreshold != 0 else 50.)
        nb.add_attribute('input_forget', lstm_params.coupledInputAndForgetGate)

        # Handle the first output of LSTM
        if lstm_params.sequenceOutput:
            lstm_y_name = context.get_unique_name('lstm_Y')
            nb.add_output(lstm_y_name)

            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [-1, hidden_size])
            post_nb1.add_input(lstm_y_name)
            post_nb1.add_output(outputs[0])

            # Handle the second output of LSTM
            if len(outputs) > 1:
                lstm_h_name = context.get_unique_name('lstm_h')
                nb.add_output(lstm_h_name)

                post_nb2 = NodeBuilder(context, 'Reshape')
                builder_list.append(post_nb2)
                post_nb2.add_attribute('shape', [1, hidden_size])
                post_nb2.add_input(lstm_h_name)
                post_nb2.add_output(outputs[1])
        else:
            # Here we ingore ONNX RNN's first output because it's useless.
            nb.add_output(context.get_unique_name('Dummy'))

            # Handle the second output of ONNX LSTM. It will become the first and the second outputs of
            # CoreML's LSTM.
            lstm_h_name = context.get_unique_name('lstm_h')
            nb.add_output(lstm_h_name)

            post_nb2 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb2)
            post_nb2.add_attribute('shape', [1, hidden_size])
            post_nb2.add_input(lstm_h_name)
            post_nb2.add_output(outputs[0])

            if len(outputs) > 1:
                post_nb3 = NodeBuilder(context, 'Identity')
                builder_list.append(post_nb3)
                post_nb3.add_input(outputs[0])
                post_nb3.add_output(outputs[1])

        # Handle the cell state output of LSTM
        if len(outputs) > 2:
            lstm_c_name = context.get_unique_name('lstm_c')
            nb.add_output(lstm_c_name)

            post_nb4 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb4)
            post_nb4.add_attribute('shape', [1, hidden_size])
            post_nb4.add_input(lstm_c_name)
            post_nb4.add_output(outputs[2])

        return [builder.make_node() for builder in builder_list]


registration.register_nn_converter('uniDirectionalLSTM', UniDirectionalLSTMLayerConverter)
