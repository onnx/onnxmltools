# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from distutils.version import StrictVersion
from .....proto import onnx_proto
from ....common._registration import register_converter
from .Reshape import apply_reshape


def extract_rnn_activation_info(activation):
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
        raise ValueError('Unsupported activation function: {}'.format(activation_type))

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


def convert_simple_rnn(scope, operator, container):
    # The RNN inputs are feature vector, X, and initial hidden state, h_init. Let C_in and C_out denote
    # the input feature length and the output dimension, respectively. Assume that S is the sequence length
    # (i.e., S-axis is the time axis of a sequence). In CorML, the shapes of X and h_init are [S, C_in] and
    # [1, C_out] respectively. In ONNX, the two shapes become [S, N, C_in] (X) and [D, N, C_out]
    # (h_init), where N is the batch size. To simulate CoreML RNN under ONNX, we need to introduce some extra
    # operators. Note that N=1 and D=1 always hold in ONNX if we are considering RNN from CoreML because there
    # is no batch size in CoreML and CoreML RNN is always uni-directional.
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
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] -----> ONNX RNN --> Y' [S, 1, C_out] --> Reshape --> Y [S, C_out]
    #                                                        ^ |
    #                                                        | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C_out]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y_h [1, C_out]
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
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] -----> ONNX RNN --> Y' [S, 1, C_out] Here Y' is useless.
    #                                                        ^ |
    #                                                        | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C_out]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
    #                                                                                                   |
    #                                                                                                   v
    #                                                                                                 Identity
    #                                                                                                   |
    #                                                                                                   v
    #                                                                                               Y_h [1, C_out]

    params = operator.raw_operator.simpleRecurrent
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    X_name = operator.inputs[0].full_name
    X_reshape_name = scope.get_unique_variable_name('X')
    apply_reshape(scope, X_name, X_reshape_name, container, desired_shape=[-1, 1, input_size])

    rnn_op_name = scope.get_unique_operator_name('RNN')
    rnn_attrs = {'name': rnn_op_name}
    rnn_inputs = [X_reshape_name]

    # Load RNN's weight matrix and add it into RNN's input list
    rnn_w_name = scope.get_unique_variable_name(rnn_op_name + '_W')
    container.add_initializer(rnn_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, hidden_size, input_size], params.weightMatrix.floatValue)
    rnn_inputs.append(rnn_w_name)

    # Load RNN's recursion matrix and add it into RNN's input list
    rnn_r_name = scope.get_unique_variable_name(rnn_op_name + '_R')
    container.add_initializer(rnn_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, hidden_size, hidden_size], params.recursionMatrix.floatValue)
    rnn_inputs.append(rnn_r_name)

    if params.hasBiasVector:
        # Load RNN's bias vector and add it into RNN's input list
        rnn_b_name = scope.get_unique_variable_name(rnn_op_name + '_B')
        rnn_b_content = np.concatenate([params.biasVector.floatValue, np.zeros(hidden_size)]).flatten()
        container.add_initializer(rnn_b_name, onnx_proto.TensorProto.FLOAT, [1, 2 * hidden_size], rnn_b_content)
        rnn_inputs.append(rnn_b_name)
    else:
        # Input names are position-sensitive, so for optional but missing inputs, we need to provide an empty string.
        rnn_inputs.append('')

    # The input, sequence_lens, in ONNX is alwasy optional for this conversion, so here is always an empty string.
    rnn_inputs.append('')

    # If initial hidden state is provided, we add it into RNN's input list after adjusting its shape.
    if len(operator.inputs) == 2:
        rnn_h_init_reshape_name = scope.get_unique_variable_name(rnn_op_name + '_h_init')
        apply_reshape(scope, operator.inputs[1].full_name, rnn_h_init_reshape_name, container,
                      desired_shape=[1, 1, hidden_size])

        rnn_inputs.append(rnn_h_init_reshape_name)
        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
    else:
        # Input names are position-sensitive, so for optional but missing inputs, we need to provide an empty string.
        rnn_inputs.append('')

    # Add RNN's information of activation function
    activation, alpha, beta = extract_rnn_activation_info(params.activation)
    rnn_attrs['activations'] = [activation.encode('ascii')]
    if alpha is not None:
        rnn_attrs['activation_alpha'] = [alpha]
    if beta is not None:
        rnn_attrs['activation_beta'] = [beta]

    # Set up other attributes
    rnn_attrs['direction'] = 'reverse' if params.reverseInput else 'forward'
    rnn_attrs['hidden_size'] = hidden_size

    # Set up version-dependent attributes
    if operator.targeted_onnx_version < StrictVersion('1.2'):
        rnn_attrs['output_sequence'] = params.sequenceOutput
        op_version = 1
    else:
        op_version = 7

    # We use the collected information to build ONNX's RNN. ONNX RNN's outputs will be saved onto two intermediate
    # tensors and we will adjust them subsequently to mimic Keras output format.
    rnn_y_name = scope.get_unique_variable_name(rnn_op_name + '_Y')
    rnn_h_name = scope.get_unique_variable_name(rnn_op_name + '_Y_h')
    container.add_node('RNN', rnn_inputs, [rnn_y_name, rnn_h_name], op_version=op_version, **rnn_attrs)

    # Set up outputs' of RNN
    if params.sequenceOutput:
        # Connect ONNX's output and CoreML's output via a reshape operator
        apply_reshape(scope, rnn_y_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

        # Handel the second RNN output (aka last hidden state), which is optional.
        if len(operator.outputs) == 2:
            # Connect ONNX's output and CoreML's output via a reshape operator
            apply_reshape(scope, rnn_h_name, operator.outputs[1].full_name, container, desired_shape=[1, hidden_size])
    else:
        # According to CoreML, its two outputs are always identical, so we just need to compute one of them and produce
        # the other one using an identity operator. Note that the first ONNX RNN output is undefined in this case.

        # Reshape last hidden state's ONNX format to its CoreML format
        apply_reshape(scope, rnn_h_name, operator.outputs[0].full_name, container, desired_shape=[1, hidden_size])

        if len(operator.outputs) == 2:
            # Copy the first output to the second output
            container.add_node('Identity', operator.outputs[0].full_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Identity'))


register_converter('simpleRecurrent', convert_simple_rnn)
