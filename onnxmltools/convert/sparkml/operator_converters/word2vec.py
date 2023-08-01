# SPDX-License-Identifier: Apache-2.0

import pandas
import numpy
from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_add, apply_mul, apply_sum
from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import StringTensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def convert_word2vec(scope, operator, container):
    op = operator.raw_operator
    vectors = (
        op.getVectors()
        .toPandas()
        .vector.apply(lambda x: pandas.Series(x.toArray()))
        .values.astype(numpy.float32)
    )
    cats_strings = op.getVectors().toPandas().word.values.tolist()
    cats_int64s = [x for x in range(0, len(cats_strings))]
    word_count = operator.inputs[0].type.shape[1]

    vectors_tensor = scope.get_unique_variable_name("vectors_tensor")
    container.add_initializer(
        vectors_tensor, onnx_proto.TensorProto.FLOAT, vectors.shape, vectors.flatten()
    )
    word_indices = scope.get_unique_variable_name("word_indices_tensor")
    container.add_node(
        "CategoryMapper",
        operator.input_full_names,
        word_indices,
        op_domain="ai.onnx.ml",
        cats_int64s=cats_int64s,
        cats_strings=cats_strings,
        default_int64=-1,
    )
    one = scope.get_unique_variable_name("one_tensor")
    container.add_initializer(one, onnx_proto.TensorProto.INT64, [1], [1])
    zero = scope.get_unique_variable_name("zero_tensor")
    container.add_initializer(zero, onnx_proto.TensorProto.INT64, [1], [0])

    sliced_outputs = []
    for i in range(0, word_count):
        index = scope.get_unique_variable_name("index_tensor")
        container.add_initializer(index, onnx_proto.TensorProto.INT64, [1], [i])
        selected_index = scope.get_unique_variable_name("selected_index_tensor")
        container.add_node(
            "ArrayFeatureExtractor",
            [word_indices, index],
            selected_index,
            op_domain="ai.onnx.ml",
        )
        reshaped_index = scope.get_unique_variable_name("reshaped_tensor")
        container.add_node(
            "Reshape", [selected_index, one], reshaped_index, op_version=5
        )
        end_index = scope.get_unique_variable_name("end_index_tensor")
        apply_add(scope, [one, reshaped_index], end_index, container, axis=0)
        sliced_output = scope.get_unique_variable_name("sliced_tensor")
        container.add_node(
            "DynamicSlice",
            [vectors_tensor, reshaped_index, end_index, zero],
            sliced_output,
        )
        sliced_outputs.append(sliced_output)

    sum_vector = scope.get_unique_variable_name("sum_tensor")
    apply_sum(scope, sliced_outputs, sum_vector, container)

    factor = scope.get_unique_variable_name("factor_tensor")
    container.add_initializer(
        factor,
        onnx_proto.TensorProto.FLOAT,
        [1],
        [1 / operator.inputs[0].type.shape[1]],
    )
    apply_mul(scope, [factor, sum_vector], operator.output_full_names, container)


register_converter("pyspark.ml.feature.Word2VecModel", convert_word2vec)


def calculate_word2vec_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[StringTensorType])

    N = operator.inputs[0].type.shape[0]
    C = operator.raw_operator.getOrDefault("vectorSize")
    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator(
    "pyspark.ml.feature.Word2VecModel", calculate_word2vec_output_shapes
)
