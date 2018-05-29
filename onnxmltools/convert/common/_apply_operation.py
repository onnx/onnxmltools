# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This file contains some high-level APIs for applying operations on variables specified by names. We should try our
# best to use those functions because they can produce ONNX operators according to the ONNX version specified in the
# `container` argument. Notice that those function behaviors are defined in a way very similar to ONNX-1.2.

from distutils.version import StrictVersion
from ...proto import onnx_proto


def _create_name_or_use_existing_one(scope, op_type, name):
    if name is None:
        return scope.get_unique_operator_name(op_type)
    else:
        return name


def _apply_unary_operation(scope, op_type, input_name, output_name, container, operator_name, **attrs):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs['name'] = name
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0]
        op_version = 1
    else:
        op_version = 6

    container.add_node(op_type, input_name, output_name, op_version=op_version, **attrs)


def _apply_basic_numerical_operation(scope, op_type, input_names, output_name, container, operator_name,
                                     axis, broadcast):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs = {}
    if container.targeted_onnx_version < StrictVersion('1.2'):
        # Before ONNX-1.2, broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs['axis'] = axis
        if broadcast is not None:
            attrs['broadcast'] = broadcast

        if container.targeted_onnx_version <= StrictVersion('1.0'):
            attrs['consumed_inputs'] = [0, 0]
            op_version = 1
        else:
            op_version = 6
    else:
        # Since ONNX-1.2, broadcasting behavior is Numpy-like, so we don't need to specify any attributes
        op_version = 7

    container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)


def _apply_pointwise_operation(scope, op_type, input_names, output_name, container, operator_name):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs = {}
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)


def apply_abs(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Abs', input_name, output_name, container, operator_name=operator_name)


def apply_add(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Add', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_batch_norm(scope, input_names, output_names, container, operator_name=None,
                     epsilon=None, is_test=None, momentum=None, spatial=None):
    name = _create_name_or_use_existing_one(scope, 'BatchNormalization', operator_name)

    attrs = {'name': name, 'epsilon': epsilon, 'is_test': is_test, 'momentum': momentum, 'spatial': spatial}

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node('BatchNormalization', input_names, output_names, op_version=op_version, **attrs)


def apply_cast(scope, input_name, output_name, container, operator_name=None, to=None):
    '''
    :param to: enum defined in ONNX TensorProto.DataType, for example, TensorProto.FLOAT and TensorProto.INT64.
    '''
    name = _create_name_or_use_existing_one(scope, 'Cast', operator_name)
    attrs = {'name': name}

    d = onnx_proto.TensorProto.DataType.DESCRIPTOR
    allowed_type_name_and_type_enum_pairs = {v.number: k for k, v in d.values_by_name.items()}
    if to not in allowed_type_name_and_type_enum_pairs:
        raise ValueError('Attribute to must be one of %s' % allowed_type_name_and_type_enum_pairs.keys())

    if container.targeted_onnx_version < StrictVersion('1.2'):
        # Convert enum to string, for example, TensorProto.INT64 to 'INT64'
        attrs['to'] = allowed_type_name_and_type_enum_pairs[to]
        op_version = 1
    else:
        # Enum, for example, TensorProto.INT64
        attrs['to'] = to
        op_version = 7

    container.add_node('Cast', input_name, output_name, op_version=op_version, **attrs)


def apply_div(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Div', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_exp(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Exp', input_name, output_name, container, operator_name=operator_name)


def apply_concat(scope, input_names, output_name, container, operator_name=None, axis=0):
    name = _create_name_or_use_existing_one(scope, 'Concat', operator_name)

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        op_version = 1
    else:
        op_version = 4

    container.add_node('Concat', input_names, output_name, op_version=op_version, name=name, axis=axis)


def apply_clip(scope, input_name, output_name, container, operator_name=None, max=None, min=None):
    name = _create_name_or_use_existing_one(scope, 'Clip', operator_name)

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


def apply_instance_norm(scope, input_names, output_name, container, operator_name=None, epsilon=1e-5):
    name = _create_name_or_use_existing_one(scope, 'InstanceNormalization', operator_name)

    attrs = {'name': name, 'epsilon': epsilon}

    if container.targeted_onnx_version <= StrictVersion('1.0'):
        attrs['consumed_inputs'] = [0] * len(input_names)
        op_version = 1
    else:
        op_version = 6

    container.add_node('InstanceNormalization', input_names, output_name, op_version=op_version, **attrs)


def apply_log(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Log', input_name, output_name, container, operator_name=operator_name)


def apply_max(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Max', input_names, output_name, container, operator_name)


def apply_mean(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Mean', input_names, output_name, container, operator_name)


def apply_min(scope, input_names, output_name, container, operator_name=None):
    _apply_pointwise_operation(scope, 'Min', input_names, output_name, container, operator_name)


def apply_mul(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    _apply_basic_numerical_operation(scope, 'Mul', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_pad(scope, input_name, output_name, container, operator_name=None, mode=None, pads=None, value=None):
    name = _create_name_or_use_existing_one(scope, 'Pad', operator_name)

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
    _apply_unary_operation(scope, 'Reciprocal', input_name, output_name, container, operator_name=operator_name)


def apply_reshape(scope, input_name, output_name, container, operator_name=None, desired_shape=None):
    name = _create_name_or_use_existing_one(scope, 'Reshape', operator_name)

    if container.targeted_onnx_version < StrictVersion('1.2'):
        container.add_node('Reshape', input_name, output_name, op_version=1, name=name, shape=desired_shape,
                           consumed_inputs=[0])
    else:
        # The shape attribute of Reshape becomes a tensor input, so we create one tensor to store that attribute.
        desired_shape_name = scope.get_unique_variable_name('shape_tensor')
        container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(desired_shape)], desired_shape)

        # Create ONNX Reshape operator
        container.add_node('Reshape', [input_name, desired_shape_name], output_name, op_version=5, name=name)


def apply_sqrt(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Sqrt', input_name, output_name, container, operator_name=operator_name)


def apply_pow(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
    name = _create_name_or_use_existing_one(scope, 'Pow', operator_name)

    attrs = {'name': name}
    if container.targeted_onnx_version < StrictVersion('1.2'):
        # Before ONNX-1.2, broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs['axis'] = axis
        if broadcast is not None:
            attrs['broadcast'] = broadcast
        op_version = 1
    else:
        # Since ONNX-1.2, broadcasting behavior is Numpy-like, so we don't need to specify any attributes
        op_version = 7
    container.add_node('Pow', input_names, output_name, op_version=op_version, **attrs)


def apply_sub(scope, input_names, output_name, container, operator_name=None, axis=None, broadcast=0):
    _apply_basic_numerical_operation(scope, 'Sub', input_names, output_name, container, operator_name=operator_name,
                                     axis=axis, broadcast=broadcast)


def apply_split(scope, input_name, output_names, container, operator_name=None, split=None, axis=0):
    name = _create_name_or_use_existing_one(scope, 'Split', operator_name)
    if container.targeted_onnx_version <= StrictVersion('1.0'):
        op_version = 1
    else:
        op_version = 2

    attrs = {'name': name}
    if split is not None:
        attrs['split'] = split
    if axis is not None:
        attrs['axis'] = axis

    container.add_node('Split', input_name, output_names, op_version=op_version, **attrs)


def apply_tile(scope, input_name, output_name, container, operator_name=None, repeats=None):
    name = _create_name_or_use_existing_one(scope, 'Split', operator_name)

    if repeats is None or all(repeat_count == 1 for repeat_count in repeats):
        container.add_node('Identity', input_name, output_name, name=name)
        return

    if container.targeted_onnx_version < StrictVersion('1.2'):
        intermediate_input_name = input_name
        intermediate_output_name = None

        for axis, repeat_count in enumerate(repeats):
            if repeat_count == 1:
                continue
            intermediate_output_name = scope.get_unique_variable_name(name + '_input')
            container.add_node('Tile', intermediate_input_name, intermediate_output_name,
                               name=name, tiles=repeat_count, axis=axis)
            intermediate_input_name = intermediate_output_name
            name = scope.get_unique_operator_name('Tile')  # Create a new name for next Tile

        # Use the last Tile name for the name of an Identity
        container.add_node('Identity', intermediate_output_name, output_name, op_version=1, name=name)
    else:
        # ONNX-1.2 has a new Tile and we use it here
        container.add_node('Tile', input_name, output_name, op_version=7, name=name, repeats=repeats)


def apply_transpose(scope, input_name, output_name, container, operator_name=None, perm=None):
    name = _create_name_or_use_existing_one(scope, 'Transpose', operator_name)

    container.add_node('Transpose', input_name, output_name, name=name, perm=perm)


def apply_upsample(scope, input_name, output_name, container, operator_name=None, mode='nearest', scales=None):
    '''
    :param mode: nearest or linear
    :param scales: an integer list of scaling-up rate of all input dimensions
    '''
    name = _create_name_or_use_existing_one(scope, 'Transpose', operator_name)

    attrs = {'name': name}
    if container.targeted_onnx_version < StrictVersion('1.2'):
        if len(scales) != 4:
            raise ValueError('Need to specify a 4-element list the the scales of N-, C-, H-, and W-axes')
        attrs['height_scale'] = float(scales[2])
        attrs['width_scale'] = float(scales[3])
        attrs['mode'] = mode.upper()
        op_version = 1
    else:
        attrs['scales'] = map(float, scales)
        attrs['mode'] = mode.lower()
        op_version = 7

    container.add_node('Upsample', input_name, output_name, op_version=op_version, **attrs)


def apply_leaky_relu(scope, input_name, output_name, container, operator_name=None, alpha=None):
    _apply_unary_operation(scope, 'LeakyRelu', input_name, output_name, container, operator_name, alpha=alpha)


def apply_relu(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Relu', input_name, output_name, container, operator_name)


def apply_prelu(scope, input_name, output_name, container, operator_name=None, slope=None):
    _apply_unary_operation(scope, 'PReLU', input_name, output_name, container, operator_name, slope=slope)


def apply_elu(scope, input_name, output_name, container, operator_name=None, alpha=None):
    _apply_unary_operation(scope, 'Elu', input_name, output_name, container, operator_name, alpha=alpha)


def apply_tanh(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Tanh', input_name, output_name, container, operator_name)


def apply_sigmoid(scope, input_name, output_name, container, operator_name=None):
    _apply_unary_operation(scope, 'Sigmoid', input_name, output_name, container, operator_name)


def apply_hard_sigmoid(scope, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
    _apply_unary_operation(scope, 'HardSigmoid', input_name, output_name, container, operator_name,
                           alpha=alpha, beta=beta)
