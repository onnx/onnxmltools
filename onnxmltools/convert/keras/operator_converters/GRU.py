import numpy as np
from keras.layers import GRU
from ....proto import onnx_proto
from ...common._registration import register_converter
from .common import extract_recurrent_activation


def convert_keras_gru(scope, operator, container):
    op = operator.raw_operator
    if hasattr(op, 'return_state') and op.return_state:
        raise RuntimeError('support state in outputs not supported')
    hidden_size = op.units
    input_size = op.input_shape[-1]
    seq_length = op.input_shape[-2]
    output_seq = op.return_sequences
    reverse_input = op.go_backwards

    op_type = 'GRU'
    attrs = {'name': operator.full_name}
    gru_input_names = []

    gru_x_name = scope.get_unique_variable_name('gru_x')
    container.add_node('Reshape', operator.inputs[0].full_name, gru_x_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1, 1, input_size])
    gru_input_names.append(gru_x_name)

    tensor_w_name = scope.get_unique_variable_name('tensor_w')
    W = op.get_weights()[0].T
    container.add_initializer(tensor_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, input_size], W.flatten())
    gru_input_names.append(tensor_w_name)

    tensor_r_name = scope.get_unique_variable_name('tensor_r')
    R = op.get_weights()[1].T
    container.add_initializer(tensor_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, hidden_size], R.flatten())
    gru_input_names.append(tensor_r_name)

    B = op.get_weights()[2]
    if op.use_bias and len(B) > 0:
        tensor_b_name = scope.get_unique_variable_name('tensor_b')
        B = np.concatenate([B, np.zeros(3 * hidden_size)])
        container.add_initializer(tensor_b_name, onnx_proto.TensorProto.FLOAT, [1, 6 * hidden_size], B.flatten())
        gru_input_names.append(tensor_b_name)
    else:
        gru_input_names.append('')

    # sequence lens
    gru_input_names.append('')
    # TODO: figure out keras way of inital_h
    gru_input_names.append('')

    activation_types = []
    alphas = []
    betas = []
    for (activation_type, alpha, beta) in \
            [extract_recurrent_activation(op.recurrent_activation), extract_recurrent_activation(op.activation)]:
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)

    attrs['activations'] = activation_types
    if alphas:
        attrs['activation_alpha'] = alphas
    if betas:
        attrs['activation_beta'] = betas

    attrs['direction'] = 'reverse' if reverse_input else 'forward'
    attrs['output_sequence'] = 1 if output_seq else 0
    attrs['hidden_size'] = hidden_size

    gru_y_name = scope.get_unique_variable_name('gru_y')
    gru_h_name = scope.get_unique_variable_name('gru_h')
    gru_output_names = [gru_y_name, gru_h_name]

    container.add_node(op_type, gru_input_names, gru_output_names, **attrs)
    if output_seq:
        intermediate_result_name = scope.get_unique_variable_name('intermediate_result')
        container.add_node('Transpose', gru_y_name, intermediate_result_name,
                           name=scope.get_unique_operator_name('Transpose'), perm=[1, 0, 2])
        container.add_node('Reshape', intermediate_result_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, seq_length, hidden_size])
    else:
        # Here we ingore ONNX GRU's first output because it's useless.
        intermediate_result_name = scope.get_unique_variable_name('intermediate_result')
        container.add_node('Transpose', gru_h_name, intermediate_result_name,
                           name=scope.get_unique_operator_name('Transpose'), perm=[1, 0, 2])
        container.add_node('Reshape', intermediate_result_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, hidden_size])


register_converter(GRU, convert_keras_gru)
