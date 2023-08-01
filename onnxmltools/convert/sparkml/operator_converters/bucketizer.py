# SPDX-License-Identifier: Apache-2.0

import copy
from onnx import onnx_pb as onnx_proto
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_bucketizer(scope, operator, container):
    op = operator.raw_operator
    splits = op.getSplits()
    if splits[0] != -float("inf") or splits[-1] != float("inf"):
        raise RuntimeError("the Splits must include positive/negative infinity")

    input_shape = operator.inputs[0].type.shape
    reshape_data = scope.get_unique_variable_name("reshape_info_tensor")
    container.add_initializer(
        reshape_data, onnx_proto.TensorProto.INT64, [3], [-1, input_shape[1], 1]
    )
    outputs = []
    for split in splits[1:]:
        less_output = scope.get_unique_variable_name("less_output_tensor")
        initializer_name = "initializer_" + str(split)
        container.add_initializer(
            initializer_name, onnx_proto.TensorProto.FLOAT, [1], [split]
        )
        container.add_node(
            "Less",
            [operator.inputs[0].full_name, initializer_name],
            less_output,
            name=scope.get_unique_operator_name("Less"),
            op_version=7,
        )
        casted_output = scope.get_unique_variable_name("cast_output_tensor")
        container.add_node(
            "Cast",
            less_output,
            casted_output,
            name=scope.get_unique_operator_name("Cast"),
            op_version=6,
            to=1,
        )
        redim_output = scope.get_unique_variable_name("reshape_output_tensor")
        container.add_node(
            "Reshape",
            [casted_output, reshape_data],
            redim_output,
            name=scope.get_unique_operator_name("Reshape"),
            op_version=5,
        )
        outputs.append(redim_output)
    concat_output = scope.get_unique_variable_name("concat_output_tensor")
    container.add_node(
        "Concat",
        outputs,
        concat_output,
        name=scope.get_unique_operator_name("Concat"),
        op_version=1,
        axis=2,
    )
    argmax_output = scope.get_unique_variable_name("argmax_output_tensor")
    container.add_node(
        "ArgMax",
        concat_output,
        argmax_output,
        name=scope.get_unique_operator_name("ArgMax"),
        op_version=1,
        axis=2,
        keepdims=0,
    )
    container.add_node(
        "Cast",
        argmax_output,
        operator.output_full_names,
        name=scope.get_unique_operator_name("Cast"),
        op_version=6,
        to=1,
    )


register_converter("pyspark.ml.feature.Bucketizer", convert_bucketizer)


def calculate_bucketizer_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType],
        good_output_types=[FloatTensorType],
    )
    input_type = copy.deepcopy(operator.inputs[0].type)
    for output in operator.outputs:
        output.type = input_type


register_shape_calculator(
    "pyspark.ml.feature.Bucketizer", calculate_bucketizer_output_shapes
)
