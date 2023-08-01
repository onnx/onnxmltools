# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .....proto import onnx_proto
from ....common._registration import register_converter
from .Reshape import apply_reshape
from .SimpleRNN import extract_rnn_activation_info


def convert_unidirectional_lstm(scope, operator, container):
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

    params = operator.raw_operator.uniDirectionalLSTM
    lstm_params = params.params
    lstm_weights = params.weightParams
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    # Initialize materials needed to create ONNX LSTM
    lstm_op_name = scope.get_unique_operator_name('LSTM')
    lstm_attrs = {'name': lstm_op_name}
    lstm_inputs = []
    lstm_outputs = []

    # Reshape input feature vector in CoreML format into ONNX format
    lstm_x_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_X_reshape')
    apply_reshape(scope, operator.inputs[0].full_name, lstm_x_reshape_name, container,
                  desired_shape=[-1, 1, input_size])
    lstm_inputs.append(lstm_x_reshape_name)

    # Allocate LSTM's weight matrices and add them into ONNX LSTM's input list
    matrices_w = np.concatenate([lstm_weights.inputGateWeightMatrix.floatValue,
                                 lstm_weights.outputGateWeightMatrix.floatValue,
                                 lstm_weights.forgetGateWeightMatrix.floatValue,
                                 lstm_weights.blockInputWeightMatrix.floatValue])
    matrices_w_name = scope.get_unique_variable_name(lstm_op_name + '_W')
    container.add_initializer(matrices_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, input_size], matrices_w)
    lstm_inputs.append(matrices_w_name)

    # Allocate LSTM's recursion weight matrices and add them into ONNX LSTM's input list
    matrices_r = np.concatenate([lstm_weights.inputGateRecursionMatrix.floatValue,
                                 lstm_weights.outputGateRecursionMatrix.floatValue,
                                 lstm_weights.forgetGateRecursionMatrix.floatValue,
                                 lstm_weights.blockInputRecursionMatrix.floatValue])
    matrices_r_name = scope.get_unique_variable_name(lstm_op_name + '_R')
    container.add_initializer(matrices_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, hidden_size], matrices_r)
    lstm_inputs.append(matrices_r_name)

    # Handle bias vectors
    vectors_b = np.zeros(shape=(8, hidden_size))
    if lstm_params.hasBiasVectors:
        vectors_b[0, :] = lstm_weights.inputGateBiasVector.floatValue
        vectors_b[1, :] = lstm_weights.outputGateBiasVector.floatValue
        vectors_b[2, :] = lstm_weights.forgetGateBiasVector.floatValue
        vectors_b[3, :] = lstm_weights.blockInputBiasVector.floatValue
    if lstm_params.forgetBias:
        # One may think we should do something like b[2, :] += 1., but it's wrong as CoreML has
        # added 1 into lstm_weights.forgetGateBiasVector.floatValue.
        pass
    if lstm_params.hasBiasVectors or lstm_params.forgetBias:
        vectors_b_name = scope.get_unique_variable_name(lstm_op_name + '_B')
        container.add_initializer(vectors_b_name, onnx_proto.TensorProto.FLOAT,
                                  [1, 8 * hidden_size], vectors_b.flatten())
        lstm_inputs.append(vectors_b_name)
    else:
        lstm_inputs.append('')

    # Converting CoreML LSTM doesn't need sequence length
    lstm_inputs.append('')

    # Provide ONNX LSTM the initial hidden state when necessary
    if len(operator.inputs) > 1:
        # Assign a Reshape to adjust CoreML hidden state's shape [1, C]/[1, C, 1, 1] into its ONNX counterpart [1, 1, C]
        lstm_h_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_h_init_reshape')
        apply_reshape(scope, operator.inputs[1].full_name, lstm_h_init_reshape_name, container,
                      desired_shape=[1, 1, hidden_size])
        lstm_inputs.append(lstm_h_init_reshape_name)

        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
    else:
        lstm_inputs.append('')

    # Provide ONNX LSTM the initial cell state when necessary
    if len(operator.inputs) > 2:
        lstm_c_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_c_init_reshape')
        apply_reshape(scope, operator.inputs[2].full_name, lstm_c_init_reshape_name, container,
                      desired_shape=[1, 1, hidden_size])
        lstm_inputs.append(lstm_c_init_reshape_name)

        # Add a zero initializer to initial cell state so that this variable becomes optional
        container.add_initializer(operator.inputs[2].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[2].type.shape,
                                  np.zeros(shape=operator.inputs[2].type.shape).flatten())
    else:
        lstm_inputs.append('')

    # Add peephole vector when presenting
    if lstm_params.hasPeepholeVectors:
        vectors_p = np.concatenate([lstm_weights.inputGatePeepholeVector.floatValue,
                                    lstm_weights.outputGatePeepholeVector.floatValue,
                                    lstm_weights.forgetGatePeepholeVector.floatValue])
        vectors_p_name = scope.get_unique_variable_name(lstm_op_name + '_P')
        container.add_initializer(vectors_p_name, onnx_proto.TensorProto.FLOAT,
                                  [1, 3 * hidden_size], vectors_p)
        lstm_inputs.append(vectors_p_name)
    else:
        lstm_inputs.append('')

    # Parse activation functions' information and add them into ONNX LSTM's attribute dictionary
    activation_types = []
    alphas = []
    betas = []
    for activation in params.activations:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode('utf-8'))
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
    lstm_attrs['direction'] = 'reverse' if params.reverseInput else 'forward'
    lstm_attrs['hidden_size'] = hidden_size
    lstm_attrs['clip'] = float(lstm_params.cellClipThreshold)
    lstm_attrs['input_forget'] = lstm_params.coupledInputAndForgetGate

    # Set up version-dependent attributes
    if container.target_opset < 7:
        lstm_attrs['output_sequence'] = lstm_params.sequenceOutput
        op_version = 1
    else:
        op_version = 7

    # Create the main LSTM operator
    lstm_y_name = scope.get_unique_variable_name(lstm_op_name + '_Y')
    lstm_y_h_name = scope.get_unique_variable_name(lstm_op_name + '_Y_h')
    lstm_c_name = scope.get_unique_variable_name(lstm_op_name + '_Y_c')
    lstm_outputs.extend([lstm_y_name, lstm_y_h_name, lstm_c_name])
    container.add_node('LSTM', lstm_inputs, lstm_outputs, op_version=op_version, **lstm_attrs)

    # Handle the first output of LSTM
    if lstm_params.sequenceOutput:
        # Handle the first output of LSTM
        apply_reshape(scope, lstm_y_name, operator.outputs[0].full_name, container, desired_shape=[-1, hidden_size])

        # Handle the second output of LSTM
        if len(operator.outputs) > 1:
            apply_reshape(scope, lstm_y_h_name, operator.outputs[1].full_name, container,
                          desired_shape=[1, hidden_size])
    else:
        # Here we ingore ONNX LSTM's first output because it's useless and use the second output of ONNX LSTM to produce
        # the first output of CoreML LSTM
        apply_reshape(scope, lstm_y_h_name, operator.outputs[0].full_name, container, desired_shape=[1, hidden_size])

        # Create the second LSTM output from the first output
        if len(operator.outputs) > 1:
            container.add_node('Identity', operator.outputs[0].full_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Identity'))

    # Handle the cell state output of LSTM
    if len(operator.outputs) > 2:
        apply_reshape(scope, lstm_c_name, operator.outputs[2].full_name, container, desired_shape=[1, hidden_size])


register_converter('uniDirectionalLSTM', convert_unidirectional_lstm)
