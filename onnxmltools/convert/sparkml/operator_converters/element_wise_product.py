# SPDX-License-Identifier: Apache-2.0

from onnx import onnx_pb as onnx_proto
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._apply_operation import apply_mul
from ...common._registration import register_converter, register_shape_calculator


def convert_element_wise_product(scope, operator, container):
    op = operator.raw_operator
    scaling_vector = scope.get_unique_variable_name('scaling_vector')
    container.add_initializer(scaling_vector, onnx_proto.TensorProto.FLOAT,
                              [1, len(op.getScalingVec())], op.getScalingVec())
    apply_mul(scope, [operator.inputs[0].full_name, scaling_vector], operator.output_full_names, container)


register_converter('pyspark.ml.feature.ElementwiseProduct', convert_element_wise_product)


def calculate_element_wise_product_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, operator.inputs[0].type.shape[1]])


register_shape_calculator('pyspark.ml.feature.ElementwiseProduct', calculate_element_wise_product_output_shapes)
