# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_reshape
from ....common._registration import register_converter


def convert_reshape_static(scope, operator, container):
    pass

    params = operator.raw_operator.reshapeStatic

    # print(params)
    intra_variable_name = operator.inputs[0].full_name

    N = operator.inputs[0].type.shape[0]
    if N == "None":
        N = -1
    if len(params.targetShape) == 4:
        output_shape = [int(d) for d in params.targetShape]
        output_shape[0] = N  # Overwrite bad default CoreML setting
    elif len(params.targetShape) == 3:
        output_shape = [N] + [int(d) for d in params.targetShape]
    elif len(params.targetShape) == 2:
        output_shape = [N] + [int(d) for d in params.targetShape]
    else:
        raise ValueError(
            "The targeted shape of Reshape (name: %s) "
            "must be 3-element or 4-element array but got %s"
            % (operator.full_name, params.targetShape)
        )

    apply_reshape(
        scope=scope,
        input_name=intra_variable_name,
        output_name=operator.outputs[0].full_name,
        container=container,
        operator_name=operator.full_name,
        desired_shape=output_shape,
    )


register_converter("reshapeStatic", convert_reshape_static)
