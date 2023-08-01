# SPDX-License-Identifier: Apache-2.0

import copy
from ...common._registration import register_converter, register_shape_calculator
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common.data_types import *


def convert_sparkml_vector_slicer(scope, operator, container):
    op = operator.raw_operator
    indices = op.getIndices()
    if not indices:
        raise ValueError("Indices are needed for conversion of VectorSlicer")

    indices_tensor = 'indices_tensor'
    container.add_initializer(indices_tensor, onnx_proto.TensorProto.INT64, [len(indices)], indices)
    container.add_node('ArrayFeatureExtractor',
                       [operator.input_full_names[0], indices_tensor], operator.output_full_names,
                       op_domain='ai.onnx.ml',
                       name=scope.get_unique_operator_name('ArrayFeatureExtractor'))


register_converter('pyspark.ml.feature.VectorSlicer', convert_sparkml_vector_slicer)


def calculate_vector_slicer_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator,
                                 good_input_types=[FloatTensorType, Int64TensorType, StringTensorType])
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape[1] = len(operator.raw_operator.getIndices())


register_shape_calculator('pyspark.ml.feature.VectorSlicer', calculate_vector_slicer_shapes)
