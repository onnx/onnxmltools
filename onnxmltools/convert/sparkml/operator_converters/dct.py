# SPDX-License-Identifier: Apache-2.0

import copy
import math
from onnx import onnx_pb as onnx_proto

import numpy

from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._apply_operation import apply_matmul, apply_mul
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_dct(scope, operator, container):
    # op = operator.raw_operator
    # inverse = op.getInverse()
    K = operator.inputs[0].type.shape[1]
    two_sqrt_n = math.sqrt(2. / K)
    sqrt_n = math.sqrt(1. / K)
    cosine_matrix = numpy.array(
        [[math.cos(math.pi * (n + .5) * k / K) for n in range(0, K)] for k in range(0, K)]
    ).transpose()
    cosine_tensor = scope.get_unique_variable_name('cosine_tensor')
    container.add_initializer(cosine_tensor, onnx_proto.TensorProto.FLOAT,
                              cosine_matrix.shape, list(cosine_matrix.flatten()))
    matmul_result = scope.get_unique_variable_name('matmul_tensor')
    apply_matmul(scope, [operator.inputs[0].full_name, cosine_tensor], matmul_result, container)
    scale_vector = [sqrt_n if (k == 0) else two_sqrt_n for k in range(0, K)]
    scale_tensor = scope.get_unique_variable_name('scale_tensor')
    container.add_initializer(scale_tensor, onnx_proto.TensorProto.FLOAT,
                              [1, len(scale_vector)], scale_vector)
    apply_mul(scope, [matmul_result, scale_tensor], operator.output_full_names, container, axis=1, broadcast=True)


register_converter('pyspark.ml.feature.DCT', convert_sparkml_dct)


def calculate_sparkml_dct_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)


register_shape_calculator('pyspark.ml.feature.DCT', calculate_sparkml_dct_output_shapes)
