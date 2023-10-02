# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers
from ...common.data_types import FloatTensorType, Int64TensorType


def convert_sparkml_vector_assembler(scope, operator, container):
    container.add_node(
        "Concat",
        [s for s in operator.input_full_names],
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("Concat"),
        op_version=4,
        axis=1,
    )


register_converter(
    "pyspark.ml.feature.VectorAssembler", convert_sparkml_vector_assembler
)


def calculate_vector_assembler_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)

    # Sum up the rank 1 (length of input vector) from each input.
    C = sum([input.type.shape[1] for input in operator.inputs])
    N = operator.inputs[0].type.shape[0]
    col_type = operator.inputs[0].type
    if isinstance(col_type, FloatTensorType):
        col_type = FloatTensorType([N, C])
    elif isinstance(col_type, Int64TensorType):
        col_type = Int64TensorType([N, C])
    else:
        raise TypeError("Unsupported input type")
    operator.outputs[0].type = col_type


register_shape_calculator(
    "pyspark.ml.feature.VectorAssembler", calculate_vector_assembler_shapes
)
