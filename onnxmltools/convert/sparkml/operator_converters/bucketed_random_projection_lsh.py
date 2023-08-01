# SPDX-License-Identifier: Apache-2.0

import numpy
from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_floor, apply_div, apply_matmul
from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from .tree_ensemble_common import save_read_sparkml_model_data

g_rand_vectors = None


def get_rand_vectors(operator):
    global g_rand_vectors
    if not g_rand_vectors:
        g_rand_vectors = (
            save_read_sparkml_model_data(
                operator.raw_params["SparkSession"], operator.raw_operator
            )
            .first()[0]
            .toArray()
            .transpose()
            .astype(numpy.float32)
        )
    return g_rand_vectors


def convert_min_hash_lsh(scope, operator, container):
    rand_vectors = get_rand_vectors(operator)
    bucket_length = float(operator.raw_operator.getOrDefault("bucketLength"))

    rand_vectors_tensor = scope.get_unique_variable_name("rand_vectors_tensor")
    container.add_initializer(
        rand_vectors_tensor,
        onnx_proto.TensorProto.FLOAT,
        rand_vectors.shape,
        rand_vectors.flatten(),
    )
    matmul_result = scope.get_unique_variable_name("matmul_result_tensor")
    apply_matmul(
        scope,
        [operator.input_full_names[0], rand_vectors_tensor],
        matmul_result,
        container,
    )
    bucket_length_tensor = scope.get_unique_variable_name("bucket_length_tensor")
    container.add_initializer(
        bucket_length_tensor, onnx_proto.TensorProto.FLOAT, [1], [bucket_length]
    )
    div_result = scope.get_unique_variable_name("div_result_tensor")
    apply_div(scope, [matmul_result, bucket_length_tensor], div_result, container)
    apply_floor(scope, div_result, operator.output_full_names[0], container)


register_converter(
    "pyspark.ml.feature.BucketedRandomProjectionLSHModel", convert_min_hash_lsh
)


def calculate_min_hash_lsh_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    N = operator.inputs[0].type.shape[0]
    C = 1
    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator(
    "pyspark.ml.feature.BucketedRandomProjectionLSHModel",
    calculate_min_hash_lsh_output_shapes,
)
