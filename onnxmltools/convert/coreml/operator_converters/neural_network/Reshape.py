# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._apply_operation import apply_transpose, apply_reshape
from ....common._registration import register_converter


def convert_reshape(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReshapeLayerParams as Params

    params = operator.raw_operator.reshape

    if params.mode == Params.CHANNEL_LAST:
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transpose')
        apply_transpose(scope, operator.inputs[0].full_name, intra_variable_name, container, perm=[0, 2, 3, 1])
    else:
        intra_variable_name = operator.inputs[0].full_name

    N = operator.inputs[0].type.shape[0]
    if N == 'None':
        N = -1
    if len(params.targetShape) == 4:
        output_shape = [int(d) for d in params.targetShape]
        output_shape[0] = N  # Overwrite bad default CoreML setting
    elif len(params.targetShape) == 3:
        output_shape = [N] + [int(d) for d in params.targetShape]
    else:
        raise ValueError('The targeted shape of Reshape (name: %s) must be 3-element or 4-element array but got %s'\
                % (operator.full_name, params.targetShape))

    apply_reshape(scope=scope, input_name=intra_variable_name, output_name=operator.outputs[0].full_name,
                  container=container, operator_name=operator.full_name, desired_shape=output_shape)


register_converter('reshape', convert_reshape)
