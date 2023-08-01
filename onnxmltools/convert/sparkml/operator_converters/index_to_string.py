# SPDX-License-Identifier: Apache-2.0

import copy

from ..utils import SparkMlConversionError
from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import Int64TensorType, StringTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def convert_index_to_string(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LabelEncoder'
    if not op.isDefined('labels') or len(op.getLabels()) == 0:
        raise SparkMlConversionError('Labels must be specified for IndexToString Transformer')
    attrs = {
        'name': scope.get_unique_operator_name(op_type),
        'classes_strings': [str(c) for c in op.getLabels()],
        'default_string': '__unknown__'
    }
    container.add_node(op_type, operator.input_full_names, operator.output_full_names,
                       op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.feature.IndexToString', convert_index_to_string)


def calculate_index_to_string_output_shapes(operator):
    '''
    This function just copy the input shape to the output because label encoder only alters input features' values, not
    their shape.
    '''
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[Int64TensorType])

    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = StringTensorType(input_shape)


register_shape_calculator('pyspark.ml.feature.IndexToString', calculate_index_to_string_output_shapes)
