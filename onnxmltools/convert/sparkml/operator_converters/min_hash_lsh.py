# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_add, apply_mul, apply_sum, apply_div, apply_sub, \
    apply_concat, apply_cast
from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ..utils import SparkMlConversionError
from .tree_ensemble_common import save_read_sparkml_model_data

MinHashLSH_HASH_PRIME = 2038074743
g_rand_coefficients = None


def get_rand_coefficients(operator):
    global g_rand_coefficients
    if not g_rand_coefficients:
        g_rand_coefficients = save_read_sparkml_model_data(
            operator.raw_params['SparkSession'], operator.raw_operator
        ).first()[0]
    return g_rand_coefficients


def convert_min_hash_lsh(scope, operator, container):
    spark = operator.raw_params['SparkSession']
    int_type = onnx_proto.TensorProto.INT64
    if spark.version < '2.4.0':
        int_type = onnx_proto.TensorProto.INT32
    rand_coefficients = get_rand_coefficients(operator)
    coeffs = []
    for i in range(0, len(rand_coefficients), 2):
        coeffs.append((rand_coefficients[i], rand_coefficients[i + 1]))
    one = scope.get_unique_variable_name('one_tensor')
    container.add_initializer(one, int_type, [1], [1])
    prime = scope.get_unique_variable_name('prime_tensor')
    container.add_initializer(prime, int_type, [1], [MinHashLSH_HASH_PRIME])

    non_zeros_int = scope.get_unique_variable_name('non_zero_int_tensor')
    container.add_node('NonZero', operator.input_full_names, non_zeros_int, op_version=9)
    non_zeros = scope.get_unique_variable_name('non_zeros_tensor')
    apply_cast(scope, non_zeros_int, non_zeros, container, to=int_type)
    remainders = []
    for coeff in coeffs:
        one_added = scope.get_unique_variable_name('one_added_tensor')
        apply_add(scope, [one, non_zeros], one_added, container)
        a = scope.get_unique_variable_name('a_coeff_tensor')
        container.add_initializer(a, int_type, [1], [coeff[0]])
        b = scope.get_unique_variable_name('b_coeff_tensor')
        container.add_initializer(b, int_type, [1], [coeff[1]])
        coeff_0_times = scope.get_unique_variable_name('coeff_0_times_tensor')
        apply_mul(scope, [a, one_added], coeff_0_times, container)
        coeff_1_added = scope.get_unique_variable_name('coeff_1_added_tensor')
        apply_add(scope, [b, coeff_0_times], coeff_1_added, container)
        # this is for Mod
        div_by_prime = scope.get_unique_variable_name('div_by_prime_tensor')
        apply_div(scope, [coeff_1_added, prime], div_by_prime, container)
        prime_x_floor = scope.get_unique_variable_name('prime_x_floor_tensor')
        apply_mul(scope, [div_by_prime, prime], prime_x_floor, container)
        remainder = scope.get_unique_variable_name('remainder_tensor')
        apply_sub(scope, [coeff_1_added, prime_x_floor], remainder, container)
        float_remainder = scope.get_unique_variable_name('float_remainder_tensor')
        apply_cast(scope, remainder, float_remainder, container, to=onnx_proto.TensorProto.FLOAT)
        min_remainder = scope.get_unique_variable_name('min_remainder_tensor')
        container.add_node('ReduceMin', float_remainder, min_remainder,
                           op_version=1,
                           axes=[1],
                           keepdims=1)
        remainders.append(min_remainder)
    apply_concat(scope, remainders, operator.output_full_names, container, axis=1)


register_converter('pyspark.ml.feature.MinHashLSHModel', convert_min_hash_lsh)


def calculate_min_hash_lsh_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    N = operator.inputs[0].type.shape[0]
    if N != 1:
        raise SparkMlConversionError('MinHashLSHModel converter cannot handle batch size of more than 1')
    C = len(get_rand_coefficients(operator)) // 2
    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('pyspark.ml.feature.MinHashLSHModel', calculate_min_hash_lsh_output_shapes)
