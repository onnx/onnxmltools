# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from keras.layers import BatchNormalization
from ...common._apply_operation import apply_batch_norm, apply_transpose
from ...common._registration import register_converter
from ....proto import onnx_proto


def convert_keras_batch_normalization(scope, operator, container):
    op = operator.raw_operator
    if op.axis != 3 and op.axis != -1:
        adjusted_input_name = operator.inputs[0].full_name
    else:
        adjusted_input_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transposed')
        apply_transpose(scope, operator.inputs[0].full_name, adjusted_input_name, container, perm=[0, 3, 1, 2])

    input_tensor_names = [adjusted_input_name]

    params = op.get_weights()
    # If scale and/or center flag is set in keras node, use keras default values for gamma and/or beta
    if not op.scale:
        params.insert(0, np.ones(params[0].shape, dtype=float))
    if not op.center:
        params.insert(1, np.zeros(params[1].shape, dtype=float))

    gamma = params[0] / np.sqrt(params[3] + op.epsilon)
    beta = params[1] - params[0] * params[2] / np.sqrt(params[3] + op.epsilon)

    scale_tensor_name = scope.get_unique_variable_name('scale')
    container.add_initializer(scale_tensor_name, onnx_proto.TensorProto.FLOAT, params[0].shape, gamma)
    input_tensor_names.append(scale_tensor_name)

    bias_tensor_name = scope.get_unique_variable_name('bias')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, params[1].shape, beta)
    input_tensor_names.append(bias_tensor_name)

    mean_tensor_name = scope.get_unique_variable_name('mean')
    container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT, params[2].shape, 0 * params[2])
    input_tensor_names.append(mean_tensor_name)

    var_tensor_name = scope.get_unique_variable_name('var')
    container.add_initializer(var_tensor_name, onnx_proto.TensorProto.FLOAT, params[3].shape, 1 + 0 * params[3])
    input_tensor_names.append(var_tensor_name)

    epsilon = op.epsilon * 1e-3  # We use a much smaller epsilon because the original epsilon is absorbed in gamma
    is_test = 1
    momentum = op.momentum
    spatial = 1

    if op.axis != 3 and op.axis != -1:
        # If no transpose is required, we can simply use the output of ONNX BatchNorm as the final outcome
        apply_batch_norm(scope, input_tensor_names, operator.output_full_names, container,
                         operator_name=operator.full_name, epsilon=epsilon, is_test=is_test,
                         momentum=momentum, spatial=spatial)
    else:
        # If transpose is required, we need to put BatchNorm's output to an intermediate tensor for applying a transpose
        intermediate_output_name = scope.get_unique_variable_name('batch_norm_output_buffer')
        apply_batch_norm(scope, input_tensor_names, intermediate_output_name, container,
                         operator_name=operator.full_name, epsilon=epsilon, is_test=is_test,
                         momentum=momentum, spatial=spatial)

        # Permute [N,C,H,W] to [N,H,W,C]
        apply_transpose(scope, intermediate_output_name, operator.outputs[0].full_name, container, perm=[0, 2, 3, 1])


register_converter(BatchNormalization, convert_keras_batch_normalization)
