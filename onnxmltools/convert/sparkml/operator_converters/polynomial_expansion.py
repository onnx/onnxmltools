# SPDX-License-Identifier: Apache-2.0

import copy
from onnx import onnx_pb as onnx_proto
from ...common._apply_operation import apply_concat, apply_cast
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_polynomial_expansion(scope, operator, container):
    op = operator.raw_operator
    feature_count = operator.inputs[0].type.shape[1]
    degree = op.getDegree()
    all_combinations = calc_combinations(feature_count, degree)
    transformed_columns = []

    for i, comb in enumerate(all_combinations):
        if comb is None:
            pass
        else:
            comb_name = scope.get_unique_variable_name("comb")
            col_name = scope.get_unique_variable_name("col")
            prod_name = scope.get_unique_variable_name("prod")

            container.add_initializer(
                comb_name, onnx_proto.TensorProto.INT64, [len(comb)], list(comb)
            )

            container.add_node(
                "ArrayFeatureExtractor",
                [operator.inputs[0].full_name, comb_name],
                col_name,
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                op_domain="ai.onnx.ml",
            )
            reduce_prod_input = col_name
            if (
                operator.inputs[0].type._get_element_onnx_type()
                == onnx_proto.TensorProto.INT64
            ):
                float_col_name = scope.get_unique_variable_name("col")
                container.add_node(
                    "Cast",
                    col_name,
                    float_col_name,
                    name=scope.get_unique_operator_name("Cast"),
                    to=onnx_proto.TensorProto.FLOAT,
                )
                reduce_prod_input = float_col_name

            container.add_node(
                "ReduceProd",
                reduce_prod_input,
                prod_name,
                axes=[1],
                name=scope.get_unique_operator_name("ReduceProd"),
            )
            transformed_columns.append(prod_name)

    if operator.inputs[0].type._get_element_onnx_type() == onnx_proto.TensorProto.INT64:
        concat_result_name = scope.get_unique_variable_name("concat_result")

        apply_concat(
            scope,
            [t for t in transformed_columns],
            concat_result_name,
            container,
            axis=1,
        )
        apply_cast(
            scope,
            concat_result_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:
        apply_concat(
            scope,
            [t for t in transformed_columns],
            operator.outputs[0].full_name,
            container,
            axis=1,
        )


register_converter(
    "pyspark.ml.feature.PolynomialExpansion", convert_sparkml_polynomial_expansion
)


def expand_inner(values, last_index, degree, multiplier, poly_values, cur_poly_index):
    if degree == 0 or last_index < 0:
        if cur_poly_index >= 0:
            # skip the very first 1
            poly_values[cur_poly_index] = multiplier.copy()
    else:
        v = values[last_index]
        last_index1 = last_index - 1
        alpha = multiplier.copy()
        i = 0
        cur_start = cur_poly_index
        while i <= degree:
            cur_start, multiplier = expand_inner(
                values, last_index1, degree - i, alpha, poly_values, cur_start
            )
            i += 1
            alpha.append(v)
    return cur_poly_index + get_poly_size(last_index + 1, degree), multiplier


def calc_combinations(feature_count, degree):
    values = range(0, feature_count)
    poly_size = get_poly_size(feature_count, degree)
    poly_values = [None] * poly_size
    expand_inner(values, feature_count - 1, degree, [], poly_values, -1)
    return poly_values


def get_poly_size(feature_count, degree):
    return get_combinations_count(feature_count + degree, degree)


def get_combinations_count(n, k):
    from math import factorial

    if n == k or k == 0:
        return 1
    if k == 1 or k == n - 1:
        return n
    return factorial(n) // factorial(k) // factorial(n - k)


def calculate_sparkml_polynomial_expansion_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType]
    )

    N = operator.inputs[0].type.shape[0]
    C = get_combinations_count(
        operator.inputs[0].type.shape[1], operator.raw_operator.getDegree()
    )
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, C]


register_shape_calculator(
    "pyspark.ml.feature.PolynomialExpansion",
    calculate_sparkml_polynomial_expansion_output_shapes,
)
