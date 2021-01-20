# SPDX-License-Identifier: Apache-2.0

from onnx import onnx_pb as onnx_proto
from ...common._registration import register_converter, register_shape_calculator
from ...common.utils import check_input_and_output_numbers
from ...common.data_types import *


def convert_sparkml_vector_indexer(scope, operator, container):
    feature_count = operator.raw_operator.numFeatures
    category_map = operator.raw_operator.categoryMaps
    split_output_names = [ scope.get_unique_variable_name('split_tensor_%d' % i) for i in range(0, feature_count)]
    if feature_count > 1:
        container.add_node('Split', operator.inputs[0].full_name, split_output_names,
                           name=scope.get_unique_operator_name('Split'),
                           op_version=2,
                           axis=1,
                           split=[1]*feature_count)
    else:
        split_output_names = operator.input_full_names
    concat_inputs = split_output_names.copy()
    for i in category_map.keys():
        converted_output = scope.get_unique_variable_name('converted_tensor_%d' % i)
        container.add_node('Cast', split_output_names[i], converted_output,
                           name=scope.get_unique_operator_name('Cast'),
                           op_version=9,
                           to=onnx_proto.TensorProto.STRING)
        attrs = {
            'name': scope.get_unique_operator_name('LabelEncoder'),
            'classes_strings': ['{0:g}'.format(c) for c in category_map[i].keys()],
            'default_string': '__unknown__'
        }
        encoded_output_name = scope.get_unique_variable_name('indexed_tensor_%d' % i)
        container.add_node('LabelEncoder', converted_output, encoded_output_name,
                       op_domain='ai.onnx.ml',
                       **attrs)
        converted_float_output = scope.get_unique_variable_name('converted_float_tensor_%d' % i)
        if feature_count == 1:
            converted_float_output = operator.output_full_names[0]
        container.add_node('Cast', encoded_output_name, converted_float_output,
                           name=scope.get_unique_operator_name('Cast'),
                           op_version=9,
                           to=onnx_proto.TensorProto.FLOAT)
        concat_inputs[i] = converted_float_output
    # add the final Concat
    if feature_count > 1:
        container.add_node('Concat', concat_inputs, operator.output_full_names[0],
                           name=scope.get_unique_operator_name('Concat'),
                           op_version=4,
                           axis=1)


register_converter('pyspark.ml.feature.VectorIndexerModel', convert_sparkml_vector_indexer)


def calculate_vector_indexer_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, operator.raw_operator.numFeatures])


register_shape_calculator('pyspark.ml.feature.VectorIndexerModel', calculate_vector_indexer_shapes)
