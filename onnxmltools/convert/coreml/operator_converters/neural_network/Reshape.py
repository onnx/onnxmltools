# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def apply_reshape(scope, input_name, output_name, desired_shape, container, operator_name=None):
    '''
    This function create a Reshape of op_version=5 to adjust the tensor indicated by input_name and put the result onto
    the tensor specified by output_name.

    :param scope: The Scope object we want to allocate this Reshape and its input
    :param input_name: A string, the name of the tensor we want to reshape
    :param output_name: A string, the name of the tensor for storing reshaped result
    :param desired_shape: A list of integers, indicating the targeted shape
    :param container: A ModelComponentContainer object used to collect all ONNX objects
    :param operator_name: The name of the ONNX Reshape we are going to create. If not specified, we may create one.
    :return:
    '''
    # The shape attribute of Reshape becomes a tensor input, so we create one tensor to store that attribute.
    desired_shape_name = scope.get_unique_variable_name('shape_tensor')
    container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(desired_shape)], desired_shape)

    # Create the name of ONNX Reshape if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Reshape')
    else:
        name = operator_name

    # Create ONNX Reshape operator
    container.add_node('Reshape', [input_name, desired_shape_name], output_name, op_version=5, name=name)

    # Return shape of the output. All other ONNX operator creator should do the same.
    return desired_shape


def convert_reshape(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReshapeLayerParams as Params

    params = operator.raw_operator.reshape

    if params.mode == Params.CHANNEL_LAST:
        op_type = 'Transpose'
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transpose')
        attrs = {'name': scope.get_unique_operator_name(op_type), 'perm': [0, 2, 3, 1]}
        container.add_node(op_type, [operator.inputs[0].full_name], [intra_variable_name], **attrs)
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
                  desired_shape=output_shape, container=container, operator_name=operator.full_name)


register_converter('reshape', convert_reshape)
