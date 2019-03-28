# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers
from onnxmltools.convert.common.data_types import *

def calculate_vector_assembler_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    C = len(operator.raw_operator.getInputCols())
    col_type = operator.inputs[0].type
    if isinstance(col_type, FloatTensorType):
        col_type = FloatTensorType([1, C])
    elif isinstance(col_type, Int64TensorType):
        col_type = Int64TensorType([1, C])
    else:
        raise TypeError("Unsupported input type")
    operator.outputs[0].type = col_type



register_shape_calculator('pyspark.ml.feature.VectorAssembler', calculate_vector_assembler_shapes)
