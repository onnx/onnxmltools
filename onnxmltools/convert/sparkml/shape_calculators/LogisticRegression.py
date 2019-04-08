# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ...common.data_types import Int64TensorType, DictionaryType, SequenceType, FloatTensorType
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types

def calculate_logistic_regression_output_shapes(operator):
    '''
     This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
     appear in this operator's output list, we should further generate a map storing all classes' probabilities.

     Allowed input/output patterns are
         1. [N, C] ---> [N, 1], A sequence of map

     '''
    class_count = operator.raw_operator.numClasses
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, class_count])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N,class_count])


register_shape_calculator('pyspark.ml.classification.LogisticRegressionModel', calculate_logistic_regression_output_shapes)
