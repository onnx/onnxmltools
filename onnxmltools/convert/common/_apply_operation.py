# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from ...proto import onnx_proto


def apply_abs(scope, input_name, output_name, container, operator_name=None):
    # Create a name if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Abs')
    else:
        name = operator_name

    attrs = {'name': name}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Abs', input_name, output_name, op_version=op_version, **attrs)


def apply_add(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Add')
    else:
        name = operator_name

    attrs = {}
    if axis is not None:
        attrs['axis'] = axis
    if broadcast is not None:
        attrs['broadcast'] = broadcast

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0, 0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Add', input_names, output_name, op_version=op_version, name=name, **attrs)


def apply_exp(scope, input_name, output_name, container, operator_name=None):
    # Create a name if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Exp')
    else:
        name = operator_name

    attrs = {'name': name}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Exp', input_name, output_name, op_version=op_version, **attrs)


def apply_clip(scope, input_name, output_name, container, operator_name=None, max=None, min=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Add')
    else:
        name = operator_name

    attrs = {'name': name}
    if max is not None:
        attrs['max'] = max
    if min is not None:
        attrs['min'] = min

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Clip', input_name, output_name, op_version=op_version, **attrs)


def apply_log(scope, input_name, output_name, container, operator_name=None):
    # Create a name if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Log')
    else:
        name = operator_name

    attrs = {'name': name}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Log', input_name, output_name, op_version=op_version, **attrs)


def apply_max(scope, input_names, output_name, container, operator_name=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Max')
    else:
        name = operator_name

    attrs = {}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node('Max', input_names, output_name, op_version=op_version, name=name, **attrs)


def apply_min(scope, input_names, output_name, container, operator_name=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Min')
    else:
        name = operator_name

    attrs = {}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node('Min', input_names, output_name, op_version=op_version, name=name, **attrs)


def apply_mul(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Mul')
    else:
        name = operator_name

    attrs = {'name': name}
    if axis is not None:
        attrs['axis'] = axis
    if broadcast is not None:
        attrs['broadcast'] = broadcast

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0, 0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Mul', input_names, output_name, op_version=op_version, **attrs)


def apply_pad(scope, input_name, output_name, container, operator_name=None, mode=None, pads=None, value=None):
    if operator_name is None:
        name = scope.get_unique_operator_name('Mul')
    else:
        name = operator_name

    attrs = {'name': name}
    if mode is not None:
        attrs['mode'] = mode
    if value is not None:
        attrs['value'] = value
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['paddings'] = pads
        op_version = 1
    else:
        attrs['pads'] = pads
        op_version = 2

    container.add_node('Pad', input_name, output_name, op_version=op_version, **attrs)


def apply_reciprocal(scope, input_name, output_name, container, operator_name=None):
    # Create a name if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Reciprocal')
    else:
        name = operator_name

    attrs = {'name': name}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Reciprocal', input_name, output_name, op_version=op_version, **attrs)


def apply_reshape(scope, input_name, output_name, container, operator_name=None, desired_shape=None):
    '''
    This function create a Reshape to adjust the tensor indicated by input_name and put the result onto the tensor
    specified by output_name.

    :param scope: The Scope object we want to allocate this Reshape and its input
    :param input_name: A string, the name of the tensor we want to reshape
    :param output_name: A string, the name of the tensor for storing reshaped result
    :param container: A ModelComponentContainer object used to collect all ONNX objects
    :param operator_name: The name of the ONNX Reshape we are going to create. If not specified, we may create one.
    :param desired_shape: A list of integers, indicating the targeted shape
    '''
    if desired_shape is None:
        raise RuntimeError('Must provide a desired shape but got None')

    # Create the name of ONNX Reshape if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Reshape')
    else:
        name = operator_name

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        container.add_node('Reshape', input_name, output_name, op_version=1, name=name, shape=desired_shape,
                           consumed_inputs=[0])
    else:
        # The shape attribute of Reshape becomes a tensor input, so we create one tensor to store that attribute.
        desired_shape_name = scope.get_unique_variable_name('shape_tensor')
        container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(desired_shape)], desired_shape)

        # Create ONNX Reshape operator
        container.add_node('Reshape', [input_name, desired_shape_name], output_name, op_version=5, name=name)


def apply_sqrt(scope, input_name, output_name, container, operator_name=None):
    # Create the name of ONNX Reshape if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Sqrt')
    else:
        name = operator_name

    attrs = {'name': name}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node('Sqrt', input_name, output_name, op_version=op_version, **attrs)


def apply_pow(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    # Create the name of ONNX Reshape if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Pow')
    else:
        name = operator_name

    attrs = {'name': name}
    if axis is not None:
        attrs['axis'] = axis
    if broadcast is not None:
        attrs['broadcast'] = broadcast

    container.add_node('Sqrt', input_names, output_name, **attrs)


def apply_transpose(scope, input_name, output_name, container, operator_name=None, perm=None):
    '''
    This function create a Transpose to adjust the tensor indicated by input_name and put the result onto the tensor
    specified by output_name.

    :param scope: The Scope object we want to allocate this Reshape and its input
    :param input_name: A string, the name of the tensor we want to reshape
    :param output_name: A string, the name of the tensor for storing reshaped result
    :param container: A ModelComponentContainer object used to collect all ONNX objects
    :param operator_name: The name of the ONNX Reshape we are going to create. If not specified, we may create one.
    :param perm: A list of integers, indicating the original coordinate indexes after permutation. For example,
    converting [N, C, H, W] to [N, H, W, C] needs to use perm=[0, 2, 3, 1].
    '''
    if perm is None:
        raise RuntimeError('Must provide perm but got None')

    # Create a name if not specified
    if operator_name is None:
        name = scope.get_unique_operator_name('Transpose')
    else:
        name = operator_name

    container.add_node('Transpose', input_name, output_name, name=name, perm=perm)
