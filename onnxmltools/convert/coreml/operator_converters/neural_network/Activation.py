# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ....common._apply_operation import (
    apply_elu,
    apply_hard_sigmoid,
    apply_leaky_relu,
    apply_prelu,
    apply_relu,
    apply_sigmoid,
    apply_tanh,
    apply_affine,
    apply_parametric_softplus,
    apply_scaled_tanh,
    apply_thresholded_relu,
)
from ....common._registration import register_converter


def convert_activation(scope, operator, container):
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {"name": operator.full_name}

    params = operator.raw_operator.activation
    activation_type = params.WhichOneof("NonlinearityType")

    # Create ONNX objects by high-level APIs such as apply_relu(...) if possible.
    # Otherwise, we use low-level APIs such as add_node(...)
    if activation_type == "leakyReLU":
        apply_leaky_relu(
            scope,
            inputs,
            outputs,
            container,
            operator_name=attrs["name"],
            alpha=params.leakyReLU.alpha,
        )
    elif activation_type == "ReLU":
        apply_relu(scope, inputs, outputs, container, operator_name=attrs["name"])
    elif activation_type == "PReLU":
        apply_prelu(
            scope,
            inputs[0],
            outputs,
            container,
            operator_name=attrs["name"],
            slope=np.asarray([params.PReLU.alpha.floatValue]),
        )
    elif activation_type == "ELU":
        apply_elu(
            scope,
            inputs,
            outputs,
            container,
            operator_name=attrs["name"],
            alpha=params.ELU.alpha,
        )
    elif activation_type == "tanh":
        apply_tanh(scope, inputs, outputs, container, operator_name=attrs["name"])
    elif activation_type == "sigmoid":
        apply_sigmoid(scope, inputs, outputs, container, operator_name=attrs["name"])
    elif activation_type == "sigmoidHard":
        apply_hard_sigmoid(
            scope,
            inputs,
            outputs,
            container,
            operator_name=attrs["name"],
            alpha=params.sigmoidHard.alpha,
            beta=params.sigmoidHard.beta,
        )
    elif activation_type == "linear":
        apply_affine(
            scope,
            inputs[0],
            outputs[0],
            container,
            operator_name=attrs["name"],
            alpha=params.linear.alpha,
            beta=params.linear.beta,
        )
    elif activation_type == "parametricSoftplus":
        apply_parametric_softplus(
            scope,
            inputs[0],
            outputs[0],
            container,
            operator_name=attrs["name"],
            alpha=params.parametricSoftplus.alpha.floatValue,
            beta=params.parametricSoftplus.beta.floatValue,
        )
    elif activation_type == "scaledTanh":
        apply_scaled_tanh(
            scope,
            inputs[0],
            outputs[0],
            container,
            operator_name=attrs["name"],
            alpha=params.scaledTanh.alpha,
            beta=params.scaledTanh.beta,
        )
    elif activation_type == "thresholdedReLU":
        apply_thresholded_relu(
            scope,
            inputs,
            outputs,
            container,
            operator_name=attrs["name"],
            alpha=params.thresholdedReLU.alpha,
        )
    else:
        if activation_type == "softsign":
            op_type = "Softsign"
        elif activation_type == "softplus":
            op_type = "Softplus"
        else:
            raise TypeError("Unsupported activation layer {0}".format(activation_type))

        container.add_node(op_type, inputs, outputs, **attrs)


register_converter("activation", convert_activation)
