# SPDX-License-Identifier: Apache-2.0

import copy

from ...common.data_types import StringTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_stop_words_remover(scope, operator, container):
    op = operator.raw_operator

    container.add_node('StringNormalizer', operator.input_full_names[0], operator.output_full_names,
                       op_domain='ai.onnx',
                       name=scope.get_unique_operator_name('StringNormalizer'),
                       op_version=10,
                       case_change_action='NONE',
                       is_case_sensitive=1 if op.getCaseSensitive() else 0,
                       stopwords=op.getStopWords())


register_converter('pyspark.ml.feature.StopWordsRemover', convert_sparkml_stop_words_remover)


def calculate_sparkml_stop_words_remover_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator,
                                 good_input_types=[StringTensorType])
    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = StringTensorType(input_shape)


register_shape_calculator('pyspark.ml.feature.StopWordsRemover', calculate_sparkml_stop_words_remover_output_shapes)
