# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .....proto import onnx_proto
from ....common._registration import register_converter
from .Reshape import apply_reshape
from .SimpleRNN import extract_rnn_activation_info


def convert_gru(scope, operator, container):
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
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX GRU --> Y' [S, 1, C_out] (this output won't be connected with others)
    #                                                    ^ |
    #                                                    | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
    #                                                                                                  |
    #                                                                                                  v
    #                                                                                               Identity
    #                                                                                                  |
    #                                                                                                  v
    #                                                                                             Y_h [1, C_out]

    params = operator.raw_operator.gru
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    # Initialize GRU's attributes. They will be used to build GRU in the end of this function.
    gru_op_name = scope.get_unique_operator_name("GRU")
    gru_attrs = {"name": gru_op_name}
    gru_inputs = []
    gru_outputs = []

    # Resahpe CoreML variable into ONNX format for feeding it into ONNX GRU
    gru_x_reshape_name = scope.get_unique_variable_name(gru_op_name + "_X_reshape")
    apply_reshape(
        scope,
        operator.inputs[0].full_name,
        gru_x_reshape_name,
        container,
        desired_shape=[-1, 1, input_size],
    )
    gru_inputs.append(gru_x_reshape_name)

    # Create weight matrices of GRU and add it into ONNX GRU's input list
    matrices_w = np.concatenate(
        [
            params.updateGateWeightMatrix.floatValue,
            params.resetGateWeightMatrix.floatValue,
            params.outputGateWeightMatrix.floatValue,
        ]
    )
    matrices_w_name = scope.get_unique_variable_name(gru_op_name + "_W")
    container.add_initializer(
        matrices_w_name,
        onnx_proto.TensorProto.FLOAT,
        [1, 3 * hidden_size, input_size],
        matrices_w,
    )
    gru_inputs.append(matrices_w_name)

    # Create recursion matrices of GRU and add it into ONNX GRU's input list
    matrices_r = np.concatenate(
        [
            params.updateGateRecursionMatrix.floatValue,
            params.resetGateRecursionMatrix.floatValue,
            params.outputGateRecursionMatrix.floatValue,
        ]
    )
    matrices_r_name = scope.get_unique_variable_name(gru_op_name + "_R")
    container.add_initializer(
        matrices_r_name,
        onnx_proto.TensorProto.FLOAT,
        [1, 3 * hidden_size, hidden_size],
        matrices_r,
    )
    gru_inputs.append(matrices_r_name)

    if params.hasBiasVectors:
        # Create bias vectors of GRU and add them into ONNX GRU's input list
        vectors_b = np.concatenate(
            [
                params.updateGateBiasVector.floatValue,
                params.resetGateBiasVector.floatValue,
                params.outputGateBiasVector.floatValue,
                np.zeros(3 * hidden_size),
            ]
        )
        vectors_b_name = scope.get_unique_variable_name(gru_op_name + "_B")
        container.add_initializer(
            vectors_b_name,
            onnx_proto.TensorProto.FLOAT,
            [1, 6 * hidden_size],
            vectors_b,
        )
        gru_inputs.append(vectors_b_name)
    else:
        # Because operator's arguments are position-sensitive, we need an empty string even if
        # this variable doesn't exist.
        gru_inputs.append("")

    # The argument, sequence length, is always missing when converting CoreML GRU.
    gru_inputs.append("")

    # Handle initial hidden state if it exists
    if len(operator.inputs) == 2:
        # Change the shape of initial state in CoreML so that ONNX's GRU is willing to take it.
        gru_h_init_reshape_name = scope.get_unique_variable_name(
            gru_op_name + "_h_init"
        )
        apply_reshape(
            scope,
            operator.inputs[1].full_name,
            gru_h_init_reshape_name,
            container,
            desired_shape=[1, 1, hidden_size],
        )
        gru_inputs.append(gru_h_init_reshape_name)

        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(
            operator.inputs[1].full_name,
            onnx_proto.TensorProto.FLOAT,
            operator.inputs[1].type.shape,
            np.zeros(shape=operator.inputs[1].type.shape).flatten(),
        )
    else:
        # Because operator's arguments are position-sensitive, we need an empty string even if
        # this variable doesn't exist.
        gru_inputs.append("")

    activation_types = []
    alphas = []
    betas = []
    for activation in params.activations:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode("utf-8"))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    gru_attrs["activations"] = activation_types
    if alphas:
        gru_attrs["activation_alpha"] = alphas
    if betas:
        gru_attrs["activation_beta"] = betas
    gru_attrs["direction"] = "reverse" if params.reverseInput else "forward"
    gru_attrs["hidden_size"] = hidden_size

    # Set up version-dependent attributes
    if container.target_opset < 5:
        gru_attrs["output_sequence"] = params.sequenceOutput
        op_version = 1
    elif container.target_opset < 7:
        gru_attrs["linear_before_reset"] = 0
        gru_attrs["output_sequence"] = params.sequenceOutput
        op_version = 3
    else:
        gru_attrs["linear_before_reset"] = 0
        op_version = 7

    # Create the major GRU operator in ONNX.
    gru_y_name = scope.get_unique_variable_name(gru_op_name + "_Y")
    gru_y_h_name = scope.get_unique_variable_name(gru_op_name + "_Y_h")
    gru_outputs.extend([gru_y_name, gru_y_h_name])
    container.add_node(
        "GRU", gru_inputs, gru_outputs, op_version=op_version, **gru_attrs
    )

    # To simulate CoreML LSTM, we add post-processing
    # operators to adjust ONNX LSTM outputs
    if params.sequenceOutput:
        # Again, the output shapes in ONNX's GRU
        # is not consistent with that in CoreML, so we need
        # to adjust the result produced by ONNX according to CoreML format.
        apply_reshape(
            scope,
            gru_y_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=[-1, hidden_size],
        )

        # Handle the second output, the last hidden state of a sequence, if exists.
        if len(operator.outputs) == 2:
            apply_reshape(
                scope,
                gru_y_h_name,
                operator.outputs[1].full_name,
                container,
                desired_shape=[1, hidden_size],
            )
    else:
        # Recall that when sequence output is false, the first and the second outputs of GRU
        # are identical. Thus, we can ignore ONNX GRU's first output.
        apply_reshape(
            scope,
            gru_y_h_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=[1, hidden_size],
        )

        if len(operator.outputs) == 2:
            container.add_node(
                "Identity",
                operator.outputs[0].full_name,
                operator.outputs[1].full_name,
                name=scope.get_unique_operator_name("Identity"),
            )


register_converter("gru", convert_gru)
