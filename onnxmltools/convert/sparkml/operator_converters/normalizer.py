# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy

from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
#from ..convert.sparkml import SparkMLConversionError
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_normalizer(scope, operator, container):
    op = operator.raw_operator
    input_name = op.getInputCol()

    op_type = 'Normalizer'
    name = scope.get_unique_operator_name(op_type)
    norm = 'L1'
    p = op.getP()
    if int(p) == 2:
        norm = 'L2'
    elif int(p) != 1:
        raise ValueError("Unsupported Norm value: " + p)
    attrs = {'name': name, 'norm': norm}
    container.add_node(op_type, input_name, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.feature.Normalizer', convert_sparkml_normalizer)


def calculate_sparkml_normalizer_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator,
                                 good_input_types=[FloatTensorType, Int64TensorType],
                                 good_output_types=[FloatTensorType])
    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = FloatTensorType(input_shape)


register_shape_calculator('pyspark.ml.feature.Normalizer', calculate_sparkml_normalizer_output_shapes)
