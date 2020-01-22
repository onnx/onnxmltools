# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ...common.data_types import StringTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_tokenizer(scope, operator, container):
    op = operator.raw_operator
    # the SPARK version converts text to lowercase and applies "\\s" regexp to it
    # Here we'll tokenize and then normalize (to convert to lowercase)
    lowercase_output = scope.get_unique_variable_name('lowercase_tensor')
    container.add_node('StringNormalizer', operator.input_full_names[0], lowercase_output,
                       op_domain='ai.onnx',
                       name=scope.get_unique_operator_name('StringNormalizer'),
                       op_version=10,
                       case_change_action='LOWER')
    container.add_node('Tokenizer', lowercase_output, operator.output_full_names,
                       op_domain='com.microsoft',
                       name=scope.get_unique_operator_name('Tokenizer'),
                       op_version=1,
                       mark=0,
                       separators=[' ', '\t', '\r', '\n'],
                       pad_value='##ERROR##',
                       mincharnum=1)


register_converter('pyspark.ml.feature.Tokenizer', convert_sparkml_tokenizer)


def calculate_sparkml_tokenizer_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator,
                                 good_input_types=[StringTensorType])
    operator.outputs[0].type = StringTensorType()


register_shape_calculator('pyspark.ml.feature.Tokenizer', calculate_sparkml_tokenizer_output_shapes)
