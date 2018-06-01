# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from distutils.version import StrictVersion
from keras.layers import LSTM
from ...common._apply_operation import apply_transpose, apply_reshape
from ...common._registration import register_converter
from ....proto import onnx_proto
from .common import extract_recurrent_activation


def convert_keras_lstm(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    input_size = op.input_shape[-1]
    seq_length = op.input_shape[-2]
    output_seq = op.return_sequences
    output_state = op.return_state
    reverse_input = op.go_backwards

    # Keras: [W_x, W_h, b] each in I F C O
    # ONNX: W[iofc] I O F C
    W_h = np.empty(shape=(4, hidden_size, hidden_size))
    W_x = np.empty(shape=(4, hidden_size, input_size))
    K_W_h = op.get_weights()[1].T
    W_h[0:] = K_W_h[0 * hidden_size:][:hidden_size]
    W_h[1:] = K_W_h[3 * hidden_size:][:hidden_size]
    W_h[2:] = K_W_h[1 * hidden_size:][:hidden_size]
    W_h[3:] = K_W_h[2 * hidden_size:][:hidden_size]

    K_W_x = op.get_weights()[0].T
    W_x[0:] = K_W_x[0 * hidden_size:][:hidden_size]
    W_x[1:] = K_W_x[3 * hidden_size:][:hidden_size]
    W_x[2:] = K_W_x[1 * hidden_size:][:hidden_size]
    W_x[3:] = K_W_x[2 * hidden_size:][:hidden_size]

    b = None
    if op.use_bias:
        b = np.zeros(shape=(8, hidden_size))
        keras_b = op.get_weights()[2]
        b[0:] = keras_b[0 * hidden_size:][:hidden_size]
        b[1:] = keras_b[3 * hidden_size:][:hidden_size]
        b[2:] = keras_b[1 * hidden_size:][:hidden_size]
        b[3:] = keras_b[2 * hidden_size:][:hidden_size]

    # Declare essential attributes of ONNX LSTM
    lstm__type = 'LSTM'
    lstm_input_names = []
    lstm_output_names = []
    lstm_attrs = {'name': operator.full_name}

    # Because of the format difference between Keras and ONNX LSTM's, we set up a preprocessing node to match them.
    lstm_x_name = scope.get_unique_variable_name('lstm_x')
    lstm_input_names.append(lstm_x_name)
    apply_reshape(scope, operator.inputs[0].full_name, lstm_x_name, container, desired_shape=[-1, 1, input_size])

    # Add the weights to the final model's initializer list so that our LSTM operator can use it
    tensor_w_name = scope.get_unique_variable_name('W')
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, input_size], W_x.flatten())
    lstm_input_names.append(tensor_w_name)

    # Add the recursion weights to the final model's initializer list so that our LSTM operator can use it
    tensor_r_name = scope.get_unique_variable_name('R')
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, hidden_size], W_h.flatten())
    lstm_input_names.append(tensor_r_name)

    if len(b) > 0:
        tensor_b_name = scope.get_unique_variable_name('B')
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 8 * hidden_size], b.flatten())
        lstm_input_names.append(tensor_b_name)
    else:
        lstm_input_names.append('')

    # sequence_lens
    lstm_input_names.append('')
    # TODO initial_h (optional) : T
    lstm_input_names.append('')
    # TODO initial_c (optional) : T
    lstm_input_names.append('')
    # P (optional) : No peep hole in keras.
    lstm_input_names.append('')

    activation_types = []
    alphas = []
    betas = []
    extracted_activations = [
        extract_recurrent_activation(op.recurrent_activation),
        extract_recurrent_activation(op.activation),
        extract_recurrent_activation(op.activation)]

    for (activation_type, alpha, beta) in extracted_activations:
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)

    lstm_attrs['activations'] = activation_types
    if alphas:
        lstm_attrs['activation_alpha'] = alphas
    if betas:
        lstm_attrs['activation_beta'] = betas

    # Set up other attributes
    lstm_attrs['direction'] = 'reverse' if reverse_input else 'forward'
    lstm_attrs['hidden_size'] = hidden_size

    # Set up version-dependent attributes
    if operator.targeted_onnx_version < StrictVersion('1.2'):
        lstm_attrs['output_sequence'] = 1 if output_seq else 0
        op_version = 1
    else:
        op_version = 7

    # We declare some names to store the outputs produced by ONNX LSTM. Then, create ONNX LSTM. Subsequently, its
    # outputs may be adjusted to match Keras format.
    lstm_y_name = scope.get_unique_variable_name('lstm_y')
    lstm_output_names.append(lstm_y_name)
    lstm_h_name = scope.get_unique_variable_name('lstm_h')
    lstm_output_names.append(lstm_h_name)
    lstm_c_name = scope.get_unique_variable_name('lstm_c')
    lstm_output_names.append(lstm_c_name)
    container.add_node(lstm__type, lstm_input_names, lstm_output_names, op_version=op_version, **lstm_attrs)

    # Create output-adjusting operators
    if output_seq:
        lstm_y_name_transposed = scope.get_unique_variable_name('lstm_y_transposed')
        apply_transpose(scope, lstm_y_name, lstm_y_name_transposed, container, perm=[1, 0, 2])
        apply_reshape(scope, lstm_y_name_transposed, operator.outputs[0].full_name, container,
                      desired_shape=[-1, seq_length, hidden_size])
    else:
        apply_reshape(scope, lstm_h_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

    if output_state:
        # state_h
        apply_reshape(scope, lstm_h_name, operator.outputs[1].full_name, container, desired_shape=[-1, hidden_size])
        # state_c
        apply_reshape(scope, lstm_c_name, operator.outputs[2].full_name, container, desired_shape=[-1, hidden_size])


register_converter(LSTM, convert_keras_lstm)
