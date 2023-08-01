# SPDX-License-Identifier: Apache-2.0

import copy
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_binarizer(scope, operator, container):
    op = operator.raw_operator
    input_name = op.getInputCol()

    op_type = "Binarizer"
    name = scope.get_unique_operator_name(op_type)
    attrs = {"name": name, "threshold": float(op.getThreshold())}
    container.add_node(
        op_type, input_name, operator.output_full_names, op_domain="ai.onnx.ml", **attrs
    )


register_converter("pyspark.ml.feature.Binarizer", convert_sparkml_binarizer)


def calculate_sparkml_binarizer_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType]
    )

    input_type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type = input_type


register_shape_calculator(
    "pyspark.ml.feature.Binarizer", calculate_sparkml_binarizer_output_shapes
)
