# SPDX-License-Identifier: Apache-2.0
import onnx


def _create_name_or_use_existing_one(scope, op_type, name):
    if name is None:
        return scope.get_unique_operator_name(op_type)
    else:
        return name


def _apply_basic_numerical_operation(
    scope, op_type, input_names, output_name, container, operator_name, axis, broadcast
):
    name = _create_name_or_use_existing_one(scope, op_type, operator_name)

    attrs = {}
    if container.target_opset < 7:
        # Before ONNX-1.2 (opset 7), broadcasting behavior is Caffe2-like.
        if axis is not None:
            attrs["axis"] = axis
        if broadcast is not None:
            attrs["broadcast"] = broadcast

        if container.target_opset < 6:
            attrs["consumed_inputs"] = [0, 0]
            op_version = 1
        else:
            op_version = 6
    else:
        # Since ONNX-1.2 (opset 7), broadcasting behavior is Numpy-like, so we don't need to specify any attributes
        op_version = 7

    container.add_node(
        op_type, input_names, output_name, op_version=op_version, name=name, **attrs
    )


def apply_div(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=None,
):
    _apply_basic_numerical_operation(
        scope,
        "Div",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_reshape(
    scope, input_name, output_name, container, operator_name=None, desired_shape=None
):
    if (
        not isinstance(desired_shape, str)
        and len(list(i for i in desired_shape if i is not None and i < 0)) > 1
    ):
        raise ValueError(
            "There can only be one -1 in the targeted shape of a Reshape but got %s"
            % desired_shape
        )

    name = _create_name_or_use_existing_one(scope, "Reshape", operator_name)

    if container.target_opset < 5:
        container.add_node(
            "Reshape",
            input_name,
            output_name,
            op_version=1,
            name=name,
            shape=desired_shape,
            consumed_inputs=[0],
        )
    else:
        if isinstance(desired_shape, str):
            desired_shape_name = desired_shape
        else:
            desired_shape_name = scope.get_unique_variable_name("shape_tensor")
            container.add_initializer(
                desired_shape_name,
                onnx.TensorProto.INT64,
                [len(desired_shape)],
                desired_shape,
            )

        # Create ONNX Reshape operator
        if isinstance(input_name, list):
            input_name.append(desired_shape_name)
        else:
            input_name = [input_name, desired_shape_name]
        container.add_node("Reshape", input_name, output_name, op_version=5, name=name)


def apply_sub(
    scope,
    input_names,
    output_name,
    container,
    operator_name=None,
    axis=None,
    broadcast=0,
):
    _apply_basic_numerical_operation(
        scope,
        "Sub",
        input_names,
        output_name,
        container,
        operator_name=operator_name,
        axis=axis,
        broadcast=broadcast,
    )


def apply_cast(scope, input_name, output_name, container, operator_name=None, to=None):
    """
    :param to: enum defined in ONNX TensorProto.DataType, for example, TensorProto.FLOAT and TensorProto.INT64.
    """
    name = _create_name_or_use_existing_one(scope, "Cast", operator_name)
    attrs = {"name": name}

    d = onnx.TensorProto.DataType.DESCRIPTOR
    allowed_type_name_and_type_enum_pairs = {
        v.number: k for k, v in d.values_by_name.items()
    }
    if to not in allowed_type_name_and_type_enum_pairs:
        raise ValueError(
            'Attribute "to" must be one of %s'
            % allowed_type_name_and_type_enum_pairs.keys()
        )

    if container.target_opset < 9:
        if to in [
            onnx.TensorProto.STRING,
            onnx.TensorProto.COMPLEX64,
            onnx.TensorProto.COMPLEX128,
        ]:
            raise ValueError(
                'Attribute "to" cannot correspond to a String or Complex TensorProto type.'
            )

        if container.target_opset < 6:
            # Convert enum to string, for example, TensorProto.INT64 to 'INT64'
            attrs["to"] = allowed_type_name_and_type_enum_pairs[to]
            op_version = 1
        else:
            # Enum, for example, TensorProto.INT64
            attrs["to"] = to
            op_version = 6
    else:
        # Enum value, for example, TensorProto.INT64
        # String casting is supported in opset 9
        if to in [onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128]:
            raise ValueError(
                'Attribute "to" cannot correspond to a Complex TensorProto type.'
            )
        attrs["to"] = to
        op_version = 9

    container.add_node("Cast", input_name, output_name, op_version=op_version, **attrs)


def apply_identity(scope, input_name, output_name, container, operator_name=None):
    name = _create_name_or_use_existing_one(scope, "Identity", operator_name)
    container.add_node("Identity", input_name, output_name, name=name)
