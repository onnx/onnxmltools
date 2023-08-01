# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .....proto import onnx_proto
from ....common._apply_operation import apply_concat, apply_split
from ....common._registration import register_converter
from .SimpleRNN import extract_rnn_activation_info
from .Reshape import apply_reshape


def convert_bidirectional_lstm(scope, operator, container):
    # The LSTM inputs are feature vector (X), initial forward hidden state (h_init), initial backward hidden
    # state (h_init_rev), initial forward cell state (c_init), and initial backward cell stat (c_init_rev)e.
    # Because of the shape differences between ONNX LSTM and CoreML LSTM, the bidirectional LSTM conversion
    # is not straightforward. See some visualizations below for details.
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

    params = operator.raw_operator.biDirectionalLSTM
    lstm_params = params.params
    lstm_weights = params.weightParams
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    lstm_op_name = scope.get_unique_operator_name("LSTM")
    lstm_attrs = {"name": lstm_op_name}
    lstm_inputs = []
    lstm_outputs = []

    lstm_x_reshape_name = scope.get_unique_variable_name(lstm_op_name + "_X_reshape")

    apply_reshape(
        scope,
        operator.inputs[0].full_name,
        lstm_x_reshape_name,
        container,
        desired_shape=[-1, 1, input_size],
    )
    lstm_inputs.append(lstm_x_reshape_name)

    # Handle LSTM's weight matrices
    matrices_w_forward = np.concatenate(
        [
            lstm_weights[0].inputGateWeightMatrix.floatValue,
            lstm_weights[0].outputGateWeightMatrix.floatValue,
            lstm_weights[0].forgetGateWeightMatrix.floatValue,
            lstm_weights[0].blockInputWeightMatrix.floatValue,
        ]
    )
    matrices_w_backward = np.concatenate(
        [
            lstm_weights[1].inputGateWeightMatrix.floatValue,
            lstm_weights[1].outputGateWeightMatrix.floatValue,
            lstm_weights[1].forgetGateWeightMatrix.floatValue,
            lstm_weights[1].blockInputWeightMatrix.floatValue,
        ]
    )
    matrices_w_name = scope.get_unique_variable_name(lstm_op_name + "_W")
    container.add_initializer(
        matrices_w_name,
        onnx_proto.TensorProto.FLOAT,
        [2, 4 * hidden_size, input_size],
        np.concatenate([matrices_w_forward, matrices_w_backward]),
    )
    lstm_inputs.append(matrices_w_name)

    # Handle LSTM's recursion matrices
    matrices_r_forward = np.concatenate(
        [
            lstm_weights[0].inputGateRecursionMatrix.floatValue,
            lstm_weights[0].outputGateRecursionMatrix.floatValue,
            lstm_weights[0].forgetGateRecursionMatrix.floatValue,
            lstm_weights[0].blockInputRecursionMatrix.floatValue,
        ]
    )
    matrices_r_backward = np.concatenate(
        [
            lstm_weights[1].inputGateRecursionMatrix.floatValue,
            lstm_weights[1].outputGateRecursionMatrix.floatValue,
            lstm_weights[1].forgetGateRecursionMatrix.floatValue,
            lstm_weights[1].blockInputRecursionMatrix.floatValue,
        ]
    )
    matrices_r_name = scope.get_unique_variable_name(lstm_op_name + "_R")
    container.add_initializer(
        matrices_r_name,
        onnx_proto.TensorProto.FLOAT,
        [2, 4 * hidden_size, hidden_size],
        np.concatenate([matrices_r_forward, matrices_r_backward]),
    )
    lstm_inputs.append(matrices_r_name)

    # Handle bias vectors
    vectors_b = np.zeros(shape=(2, 8, hidden_size))
    if lstm_params.hasBiasVectors:
        vectors_b[0, 0, :] = lstm_weights[0].inputGateBiasVector.floatValue
        vectors_b[0, 1, :] = lstm_weights[0].outputGateBiasVector.floatValue
        vectors_b[0, 2, :] = lstm_weights[0].forgetGateBiasVector.floatValue
        vectors_b[0, 3, :] = lstm_weights[0].blockInputBiasVector.floatValue
        vectors_b[1, 0, :] = lstm_weights[1].inputGateBiasVector.floatValue
        vectors_b[1, 1, :] = lstm_weights[1].outputGateBiasVector.floatValue
        vectors_b[1, 2, :] = lstm_weights[1].forgetGateBiasVector.floatValue
        vectors_b[1, 3, :] = lstm_weights[1].blockInputBiasVector.floatValue
    if lstm_params.forgetBias:
        # One may think we should do something like b[0, 2, :] += 1. and b[1, 2, :] += 1.,
        # but it's not correct as CoreML has added 1 into those bias vectors.
        pass
    if lstm_params.hasBiasVectors or lstm_params.forgetBias:
        vectors_b_name = scope.get_unique_variable_name(lstm_op_name + "_B")
        container.add_initializer(
            vectors_b_name,
            onnx_proto.TensorProto.FLOAT,
            [2, 8 * hidden_size],
            vectors_b.flatten(),
        )
        lstm_inputs.append(vectors_b_name)
    else:
        lstm_inputs.append("")

    # Due to the position sensitivity in ONNX argument parsing,
    # we add an empty string for the non-existing
    # sequence length
    lstm_inputs.append("")

    # Handle initial hidden state if necessary
    if len(operator.inputs) > 1:
        lstm_h_init_name = scope.get_unique_variable_name(lstm_op_name + "_h_init")
        apply_concat(
            scope,
            [operator.inputs[1].full_name, operator.inputs[3].full_name],
            lstm_h_init_name,
            container,
            axis=0,
        )

        lstm_h_init_reshape_name = scope.get_unique_variable_name(
            lstm_op_name + "_h_init_reshape"
        )
        apply_reshape(
            scope,
            lstm_h_init_name,
            lstm_h_init_reshape_name,
            container,
            desired_shape=[2, 1, hidden_size],
        )

        # Add zero initializers to forward and backward initial
        # hidden states so that they become optional
        container.add_initializer(
            operator.inputs[1].full_name,
            onnx_proto.TensorProto.FLOAT,
            operator.inputs[1].type.shape,
            np.zeros(shape=operator.inputs[1].type.shape).flatten(),
        )
        container.add_initializer(
            operator.inputs[3].full_name,
            onnx_proto.TensorProto.FLOAT,
            operator.inputs[3].type.shape,
            np.zeros(shape=operator.inputs[3].type.shape).flatten(),
        )
        lstm_inputs.append(lstm_h_init_reshape_name)
    else:
        lstm_inputs.append("")

    # Handle initial cell state if needed
    if len(operator.inputs) > 2:
        lstm_c_init_name = scope.get_unique_variable_name(lstm_op_name + "_c_init")
        apply_concat(
            scope,
            [operator.inputs[2].full_name, operator.inputs[4].full_name],
            lstm_c_init_name,
            container,
            axis=0,
        )

        # Reshape the Cell state so that ONNX LSTM can accept it
        lstm_c_init_reshape_name = scope.get_unique_variable_name(
            lstm_op_name + "_c_init_reshape"
        )
        apply_reshape(
            scope,
            lstm_c_init_name,
            lstm_c_init_reshape_name,
            container,
            desired_shape=[2, 1, hidden_size],
        )
        lstm_inputs.append(lstm_c_init_reshape_name)

        # Add zero initializers to forward and backward initial
        # cell states so that they become optional
        container.add_initializer(
            operator.inputs[2].full_name,
            onnx_proto.TensorProto.FLOAT,
            operator.inputs[2].type.shape,
            np.zeros(shape=operator.inputs[2].type.shape).flatten(),
        )
        container.add_initializer(
            operator.inputs[4].full_name,
            onnx_proto.TensorProto.FLOAT,
            operator.inputs[4].type.shape,
            np.zeros(shape=operator.inputs[4].type.shape).flatten(),
        )
    else:
        lstm_inputs.append("")

    # Handle peephole vectors if necessary
    if lstm_params.hasPeepholeVectors:
        p_forward = np.concatenate(
            [
                lstm_weights[0].inputGatePeepholeVector.floatValue,
                lstm_weights[0].outputGatePeepholeVector.floatValue,
                lstm_weights[0].forgetGatePeepholeVector.floatValue,
            ]
        )
        p_backward = np.concatenate(
            [
                lstm_weights[1].inputGatePeepholeVector.floatValue,
                lstm_weights[1].outputGatePeepholeVector.floatValue,
                lstm_weights[1].forgetGatePeepholeVector.floatValue,
            ]
        )
        p_name = scope.get_unique_variable_name(lstm_op_name + "_P")
        container.add_initializer(
            p_name,
            onnx_proto.TensorProto.FLOAT,
            [2, 3 * hidden_size],
            np.concatenate([p_forward, p_backward]),
        )
        lstm_inputs.append(p_name)
    else:
        lstm_inputs.append("")

    # Parse activation functions and add them into ONNX LSTM's attribute dictionary
    activation_types = []
    alphas = []
    betas = []
    for activation in params.activationsForwardLSTM:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode("utf-8"))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    for activation in params.activationsBackwardLSTM:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode("utf-8"))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    lstm_attrs["activations"] = activation_types
    if alphas:
        lstm_attrs["activation_alpha"] = alphas
    if betas:
        lstm_attrs["activation_beta"] = betas

    # Add more attributes
    lstm_attrs["direction"] = "bidirectional"
    lstm_attrs["hidden_size"] = hidden_size
    lstm_attrs["clip"] = float(lstm_params.cellClipThreshold)
    lstm_attrs["input_forget"] = lstm_params.coupledInputAndForgetGate

    # Set up version-dependent attributes
    if container.target_opset < 7:
        lstm_attrs["output_sequence"] = lstm_params.sequenceOutput
        op_version = 1
    else:
        op_version = 7

    # Create the major ONNX LSTM operator. We assign a tensor name
    # to each output of LSTM. However, variables can be
    # undefined in some cases. For example, when output_sequence=False,
    # the first output is not meaningful.
    lstm_y_name = scope.get_unique_variable_name(lstm_op_name + "_Y")
    lstm_y_h_name = scope.get_unique_variable_name(lstm_op_name + "_Y_h")
    lstm_y_c_name = scope.get_unique_variable_name(lstm_op_name + "_Y_c")
    lstm_outputs.extend([lstm_y_name, lstm_y_h_name, lstm_y_c_name])
    container.add_node(
        "LSTM", lstm_inputs, lstm_outputs, op_version=op_version, **lstm_attrs
    )

    # Create post-processing operators for converting ONNX LSTM outputs to CoreML ones
    if lstm_params.sequenceOutput:
        apply_reshape(
            scope,
            lstm_y_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=[-1, 2 * hidden_size],
        )

        if len(operator.outputs) > 1:
            lstm_y_h_reshape_name = scope.get_unique_variable_name(
                lstm_op_name + "_Y_h_reshape"
            )
            apply_reshape(
                scope,
                lstm_y_h_name,
                lstm_y_h_reshape_name,
                container,
                desired_shape=[2, hidden_size],
            )

            apply_split(
                scope,
                lstm_y_h_reshape_name,
                [operator.outputs[1].full_name, operator.outputs[3].full_name],
                container,
                split=[1, 1],
                axis=0,
            )
    else:
        # Here we ignore ONNX RNN's first output because it's useless.
        # The second output of ONNX LSTM will be used to
        # generate the first and the second outputs of CoreML LSTM.

        # Directly reshape ONNX LSTM's 2nd output to CoreML LSTM's 1st output.
        apply_reshape(
            scope,
            lstm_y_h_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=[1, 2 * hidden_size],
        )

        if len(operator.outputs) > 1:
            lstm_y_reshape_name = scope.get_unique_variable_name(
                lstm_op_name + "_Y_reshape"
            )
            apply_reshape(
                scope,
                lstm_y_h_name,
                lstm_y_reshape_name,
                container,
                desired_shape=[2, hidden_size],
            )

            apply_split(
                scope,
                lstm_y_reshape_name,
                [operator.outputs[1].full_name, operator.outputs[3].full_name],
                container,
                split=[1, 1],
                axis=0,
            )

    # Output cell state if necessary
    if len(operator.outputs) > 2:
        lstm_y_c_reshape_name = scope.get_unique_variable_name(
            lstm_op_name + "_Y_c_reshape"
        )
        apply_reshape(
            scope,
            lstm_y_c_name,
            lstm_y_c_reshape_name,
            container,
            desired_shape=[2, hidden_size],
        )

        apply_split(
            scope,
            lstm_y_c_reshape_name,
            [operator.outputs[2].full_name, operator.outputs[4].full_name],
            container,
            split=[1, 1],
            axis=0,
        )


register_converter("biDirectionalLSTM", convert_bidirectional_lstm)
