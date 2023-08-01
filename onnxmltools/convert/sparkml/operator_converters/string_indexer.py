# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List

from pyspark import SparkContext
from pyspark.ml.feature import StringIndexerModel

from ...common._registration import register_converter, register_shape_calculator
from ...common._topology import ModelComponentContainer, Operator, Scope, Variable
from ...common.data_types import Int64TensorType, StringTensorType
from ...common.utils import check_input_and_output_types


def convert_sparkml_string_indexer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op: StringIndexerModel = operator.raw_operator
    op_domain = "ai.onnx.ml"
    op_version = 2
    op_type = "LabelEncoder"

    labelsArray: List[List[str]]

    if SparkContext._active_spark_context.version.startswith("2."):
        labelsArray = [op.labels]
    else:
        labelsArray = op.labelsArray

    for i in range(0, len(labelsArray)):
        encoder_attrs = {
            "name": scope.get_unique_operator_name("StringIndexer_" + str(i)),
            "keys_strings": labelsArray[i],
            "values_int64s": list(range(0, len(labelsArray[i]))),
        }

        container.add_node(
            op_type,
            [operator.inputs[i].full_name],
            [operator.outputs[i].full_name],
            op_domain=op_domain,
            op_version=op_version,
            **encoder_attrs,
        )


register_converter(
    "pyspark.ml.feature.StringIndexerModel", convert_sparkml_string_indexer
)


def calculate_sparkml_string_indexer_output_shapes(operator: Operator):
    """
    This function just copy the input shape to the output
    because label encoder only alters input features' values, not
    their shape.
    """
    check_input_and_output_types(
        operator, good_input_types=[Int64TensorType, StringTensorType]
    )
    input: Variable
    output: Variable
    for input, output in zip(operator.inputs, operator.outputs):
        input_shape = copy.deepcopy(input.type.shape)
        output.type = Int64TensorType(input_shape)


register_shape_calculator(
    "pyspark.ml.feature.StringIndexerModel",
    calculate_sparkml_string_indexer_output_shapes,
)
