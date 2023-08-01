# SPDX-License-Identifier: Apache-2.0

from onnx import onnx_pb as onnx_proto
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._apply_operation import apply_matmul
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_pca(scope, operator, container):
    op = operator.raw_operator
    pc_tensor = scope.get_unique_variable_name('pc_tensor')
    container.add_initializer(pc_tensor, onnx_proto.TensorProto.FLOAT,
                              [op.pc.numRows, op.pc.numCols], list(op.pc.toArray().flatten()))
    apply_matmul(scope, [operator.inputs[0].full_name, pc_tensor], operator.output_full_names, container)


register_converter('pyspark.ml.feature.PCAModel', convert_sparkml_pca)


def calculate_sparkml_pca_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, operator.raw_operator.getOrDefault('k')])


register_shape_calculator('pyspark.ml.feature.PCAModel', calculate_sparkml_pca_output_shapes)
