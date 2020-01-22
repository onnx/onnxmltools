# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy

from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_imputer(scope, operator, container):
    op = operator.raw_operator

    op_type = 'Imputer'
    name = scope.get_unique_operator_name(op_type)
    attrs = {'name': name}
    input_type = operator.inputs[0].type
    surrogates = op.surrogateDF.toPandas().values[0].tolist()
    value = op.getOrDefault('missingValue')
    if isinstance(input_type, FloatTensorType):
        attrs['imputed_value_floats'] = surrogates
        attrs['replaced_value_float'] = value
    elif isinstance(input_type, Int64TensorType):
        attrs['imputed_value_int64s'] = [int(x) for x in surrogates]
        attrs['replaced_value_int64'] = int(value)
    else:
        raise RuntimeError("Invalid input type: " + input_type)

    if len(operator.inputs) > 1:
        concatenated_output = scope.get_unique_variable_name('concat_tensor')
        container.add_node('Concat', operator.input_full_names, concatenated_output,
                           name=scope.get_unique_operator_name('Concat'),
                           op_version=4,
                           axis=1)
        imputed_output = scope.get_unique_variable_name('imputed_tensor')
        container.add_node(op_type, concatenated_output, imputed_output, op_domain='ai.onnx.ml', **attrs)
        container.add_node('Split', imputed_output, operator.output_full_names,
                           name=scope.get_unique_operator_name('Split'),
                           op_version=2,
                           axis=1,
                           split=range(1, len(operator.output_full_names)))
    else:
        container.add_node(op_type, operator.inputs[0].full_name, operator.output_full_names[0],
                           op_domain='ai.onnx.ml',
                           **attrs)


register_converter('pyspark.ml.feature.ImputerModel', convert_imputer)

def calculate_imputer_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=[1, len(operator.outputs)])
    check_input_and_output_types(operator,
                                 good_input_types=[FloatTensorType, Int64TensorType],
                                 good_output_types=[FloatTensorType, Int64TensorType])
    input_type = copy.deepcopy(operator.inputs[0].type)
    for output in operator.outputs:
        output.type = input_type


register_shape_calculator('pyspark.ml.feature.ImputerModel', calculate_imputer_output_shapes)