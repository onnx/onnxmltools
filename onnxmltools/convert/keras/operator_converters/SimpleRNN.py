import numpy as np
from keras.layers import SimpleRNN
from ....proto import onnx_proto
from ...common._registration import register_converter
from .common import extract_recurrent_activation


def convert_keras_simple_rnn(scope, operator, container):
    op = operator.raw_operator
    hidden_size = op.units
    input_size = op.input_shape[-1]
    seq_length = op.input_shape[-2]
    output_seq = op.return_sequences
    reverse_input = op.go_backwards

    attrs = {'name': operator.full_name}
    rnn_input_names = []
    rnn_output_names = []

    rnn_x_name = scope.get_unique_variable_name('rnn_x')
    container.add_node('Transpose', operator.inputs[0].full_name, rnn_x_name,
                       name=scope.get_unique_operator_name('Transpose'), perm=[1, 0, 2])
    rnn_input_names.append(rnn_x_name)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    W = op.get_weights()[0].T
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, input_size], W.flatten())
    rnn_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    R = op.get_weights()[1].T
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT, [1, hidden_size, hidden_size], R.flatten())
    rnn_input_names.append(tensor_r_name)

    if op.use_bias:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        B = np.concatenate([op.get_weights()[2], np.zeros(hidden_size)])
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 2 * hidden_size], B.flatten())
        rnn_input_names.append(tensor_b_name)
    else:
        rnn_input_names.append('')

    # sequence_lens is not able to be converted from input_length
    rnn_input_names.append('')
    # TODO: figure out keras way of inital_h
    rnn_input_names.append('')

    if hasattr(op, 'activation'):
        activation_type, alpha, beta = extract_recurrent_activation(op.activation)
        attrs['activations'] = [activation_type.encode('ascii')]
        if alpha is not None:
            attrs['activation_alpha'] = [alpha]
        if beta is not None:
            attrs['activation_beta'] = [beta]

    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['output_sequence'] = 1 if output_seq else 0
    attrs['hidden_size'] = hidden_size

    rnn_y_name = scope.get_unique_variable_name('rnn_y')
    rnn_h_name = scope.get_unique_variable_name('rnn_h')
    rnn_output_names.append(rnn_y_name)
    rnn_output_names.append(rnn_h_name)
    container.add_node('RNN', rnn_input_names, rnn_output_names, **attrs)

    if output_seq:
        permuted_rnn_y_name = scope.get_unique_variable_name('rnn_y_permuted')
        container.add_node('Transpose', rnn_y_name, permuted_rnn_y_name,
                           name=scope.get_unique_operator_name('Transpose'), perm=[1, 0, 2])
        container.add_node('Reshape', permuted_rnn_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, seq_length, hidden_size])
    else:
        container.add_node('Reshape', rnn_h_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, hidden_size])
        # Here we ingore ONNX RNN's first output because it's useless.


register_converter(SimpleRNN, convert_keras_simple_rnn)
