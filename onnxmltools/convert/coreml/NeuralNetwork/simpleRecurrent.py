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


def extract_activation_info(activation):
    activation_type = activation.WhichOneof('NonlinearityType')
    alpha = None
    beta = None

    activation_map = {'linear': 'Affine',
                      'ReLU': 'Relu',
                      'leakyReLU': 'LeakyRelu',
                      'thresholdedReLU': 'ThresholdedRelu',
                      'PReLU': 'PRelu',
                      'tanh': 'Tanh',
                      'scaledTanh': 'ScaledTanh',
                      'sigmoid': 'Sigmoid',
                      'sigmoidHard': 'HardSigmoid',
                      'ELU': 'Elu',
                      'softsign': 'Softsign',
                      'softplus': 'Softplus',
                      'parametricSoftplus': 'ParametricSoftplus'}

    if activation_type not in activation_map:
        raise ValueError(
            'Unsupported activation function: {}'.format(activation_type))

    # Notice that if we see a default vaulue (i.e., 0 for float), we may replace it with
    # the real default parameter for the specified activation function if necessary.
    if activation_type == 'leakyReLU':
        alpha = activation.leakyReLU.alpha
        if alpha == 0:
            alpha = 0.3
    elif activation_type == 'PReLU':
        raise RuntimeError('Unsupported activation function: {}'.format(activation_type))
    elif activation_type == 'ELU':
        alpha = activation.ELU.alpha
    elif activation_type == 'thresholdedReLU':
        alpha = activation.thresholdedReLU.alpha
        if alpha == 0:
            alpha = 1.0
    elif activation_type == 'scaledTanh':
        alpha = activation.scaledTanh.alpha
        beta = activation.scaledTanh.beta
    elif activation_type == 'linear':
        alpha = activation.linear.alpha
        beta = activation.linear.beta
        if alpha == 0:
            alpha = 1.0
    elif activation_type == 'sigmoidHard':
        alpha = activation.sigmoidHard.alpha
        beta = activation.sigmoidHard.beta
        if alpha == 0:
            alpha = 0.2
        if beta == 0:
            beta = 0.5
    elif activation_type == 'parametricSoftplus':
        raise RuntimeError('Unsupported activation function: {}'.format(activation_type))

    return activation_map[activation_type], alpha, beta


def extract_dims(tensor):
    return [d.dim_param if d.dim_param != '' else d.dim_value for d in tensor.type.tensor_type.shape.dim]


class SimpleRecurrentLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'simpleRecurrent')
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
            onnx_name = context.get_onnx_name(top_level_input.name)
            if onnx_name == inputs[1] and top_level_input.name:
                # The shape is [C] in CoreML and loaded as ['None', C]
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
            onnx_name = context.get_onnx_name(top_level_output.name)
            if onnx_name == outputs[1] and top_level_output.name:
                # The shape is [C] in CoreML and loaded as ['None', C]
                dims = extract_dims(top_level_output)
            else:
                continue
            dims[0] = 1
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    top_level_output.type.tensor_type.shape.dim[i].dim_param = d
                else:
                    top_level_output.type.tensor_type.shape.dim[i].dim_value = d

        # The RNN inputs are feature vector, X, and initial hidden state, h_init. Let C_in and C_out denote
        # the input feature length and the output dimension, respectively. Assume that S is the sequence length
        # (i.e., S-axis is the time axis of a sequence). In CorML, the shapes of X and h_init are [S, C_in] and
        # [1, C_out] respectively. In ONNX, the two shapes become [S, N, C_in] (X) and [D, N, C_out]
        # (h_init), where N is the batch size and S is the sequence length (i.e., S-axis is the time axis). To
        # simulate CoreML RNN under ONNX, we need to introduce some extra operators. Note that N=1 and D=1 always
        # hold in ONNX if we are considering RNN from CoreML because there is no batch size in CoreML and CoreML RNN
        # is always uni-directional.
        # 
        # Below we provide a visualization of our conversion for CoreML RNN.
        #
        # Symbols:
        #  
        #  X: input features of CoreML RNN
        #  h_init: initial RNN state of CoreML
        #  Y: CoreML RNN's output. It can be [S, C_out] (if sequence_output is on ) or [1, C_out] (if sequence_output is off)
        #  Y_h: CoreML RNN's last hidden state
        #
        #  X': input features of ONNX RNN
        #  h_init': initial RNN state of ONNX
        #  Y': ONNX RNN's output
        #  Y_h': ONNX RNN's last hidden state
        #
        # Computational graph of CoreML RNN (sequence_output is on):
        #
        # X [S, C_in] ---> CoreML RNN ---> Y [S, C_out]
        #                    ^     |
        #                    |     |
        # h_init [1, C_out] -'     '---> Y_h [1, C_out]
        #
        # Computational graph we use for represent CoreML RNN into ONNX (sequence_output is on):
        #
        # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX RNN --> Y' [S, 1, C_out] --> Reshape --> Y [S, C_out]
        #                                                    ^ |
        #                                                    | |
        # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y_h [1, C_out]
        #
        # Computational graph of CoreML RNN (sequence_output is off):
        #
        # X [S, C_in] ---> CoreML RNN ---> Y [1, C_out]
        #                    ^     |
        #                    |     |
        # h_init [1, C_cou] -'     '---> Y_h [1, C_out] Note that in this case, Y=Y_h.
        #
        # Computational graph we use to represent CoreML RNN into ONNX (sequence_output is off):
        #
        # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX RNN --> Y' [S, 1, C_out] (Y' won't be connected with others)
        #                                                    ^ |
        #                                                    | |
        # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
        #                                                                                                  |
        #                                                                                                  v
        #                                                                                               Identity
        #                                                                                                  |
        #                                                                                                  v
        #                                                                                             Y_h [1, C_out]

        params = cm_node.simpleRecurrent
        input_size = params.inputVectorSize
        hidden_size = params.outputVectorSize

        builder_list = []

        rnn_x_name = context.get_unique_name('rnn_x')
        pre_nb1 = NodeBuilder(context, 'Reshape')
        builder_list.append(pre_nb1)
        pre_nb1.add_attribute('shape', [-1, 1, input_size])
        pre_nb1.add_input(inputs[0])
        pre_nb1.add_output(rnn_x_name)

        nb = NodeBuilder(context, 'RNN')
        builder_list.append(nb)

        # Use only the first input
        nb.add_input(rnn_x_name)

        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT,
                                          [1, hidden_size, input_size],
                                          params.weightMatrix.floatValue)
        nb.add_initializer(tensor_w)

        tensor_r = model_util.make_tensor('R', onnx_proto.TensorProto.FLOAT,
                                          [1, hidden_size, hidden_size],
                                          params.recursionMatrix.floatValue)
        nb.add_initializer(tensor_r)

        if params.hasBiasVector:
            bias_vector = np.concatenate([params.biasVector.floatValue,
                                          np.zeros(hidden_size)])
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT,
                                              [1, 2 * hidden_size], bias_vector.flatten())
            nb.add_initializer(tensor_b)
        else:
            nb.add_empty_input()

        # sequence_lens
        nb.add_empty_input()

        # initial_h
        if len(inputs) == 2:
            rnn_h_init_name = context.get_unique_name('rnn_h_init')
            pre_nb2 = NodeBuilder(context, 'Reshape')
            builder_list.append(pre_nb2)
            pre_nb2.add_attribute('shape', [1, 1, hidden_size])
            zero_initializer = model_util.make_tensor('h_init', onnx_proto.TensorProto.FLOAT,
                                                      [1, hidden_size], [0.] * hidden_size)
            pre_nb2.add_initializer(zero_initializer, inputs[1])
            pre_nb2.add_output(rnn_h_init_name)

            nb.add_input(rnn_h_init_name)
        else:
            nb.add_empty_input()

        activation, alpha, beta = extract_activation_info(params.activation)
        nb.add_attribute('activations', [activation.encode('ascii')])
        if alpha is not None:
            nb.add_attribute('activation_alpha', [alpha])
        if beta is not None:
            nb.add_attribute('activation_beta', [beta])

        nb.add_attribute('direction', 'reverse' if params.reverseInput else 'forward')
        nb.add_attribute('output_sequence', params.sequenceOutput)
        nb.add_attribute('hidden_size', params.outputVectorSize)

        if params.sequenceOutput:
            rnn_y_name = context.get_unique_name('rnn_Y')
            nb.add_output(rnn_y_name)
            post_nb1 = NodeBuilder(context, 'Reshape')
            builder_list.append(post_nb1)
            post_nb1.add_attribute('shape', [-1, hidden_size])
            post_nb1.add_input(rnn_y_name)
            post_nb1.add_output(outputs[0])

            if len(outputs) == 2:
                rnn_h_name = context.get_unique_name('rnn_h')
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
            rnn_h_name = context.get_unique_name('rnn_h')
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


registration.register_nn_converter('simpleRecurrent', SimpleRecurrentLayerConverter)
