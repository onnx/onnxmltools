# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

def get_permutation_config(n_dims):
    input_perm_axes = [0, n_dims + 1] + list(range(1, n_dims + 1))
    output_perm_axes = [0] + list(range(2, n_dims + 2)) + [1]
    return input_perm_axes, output_perm_axes


def extract_recurrent_activation(activation):
    from keras import activations
    alpha = None
    beta = None
    if activation == activations.sigmoid:
        onnx_op_type = 'Sigmoid'
    elif activation == activations.hard_sigmoid:
        onnx_op_type = 'HardSigmoid'
        alpha = 0.2
        beta = 0.5
    elif activation == activations.tanh:
        onnx_op_type = 'Tanh'
    elif activation == activations.relu:
        onnx_op_type = 'Relu'
    elif activation == activations.linear:
        onnx_op_type = 'Affine'
        alpha = 1.0
    else:
        raise NotImplementedError('The activation %s not supported' % activation)

    return (onnx_op_type, alpha, beta)
