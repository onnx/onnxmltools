# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy

from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType
from onnxmltools.convert.common.utils import check_input_and_output_numbers, check_input_and_output_types
#from onnxmltools.convert.sparkml import SparkMLConversionError
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_ngram(scope, operator, container):
    op = operator.raw_operator

    container.add_node('TfIdfVectorizer', operator.input_full_names[0], operator.output_full_names,
                       op_domain='ai.onnx',
                       name=scope.get_unique_operator_name('TfIdfVectorizer'),
                       op_version=9,
                       min_gram_length=op.getN(),
                       max_gram_length=op.getN(),
                       mode='TF',
                       max_skip_count=0,
                       pool_strings=['x','y'],
                       ngram_counts=[0, 0],
                       ngram_indexes=[0])


register_converter('pyspark.ml.feature.NGram', convert_sparkml_ngram)


def calculate_sparkml_ngram_output_shapes(operator):
    pass
    # check_input_and_output_numbers(operator, output_count_range=1)
    # check_input_and_output_types(operator,
    #                              good_input_types=[FloatTensorType, Int64TensorType],
    #                              good_output_types=[FloatTensorType])
    # input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    # operator.outputs[0].type = FloatTensorType(input_shape)


register_shape_calculator('pyspark.ml.feature.NGram', calculate_sparkml_ngram_output_shapes)
