# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common.data_types import (
    FloatTensorType, Int64TensorType,
    StringTensorType,
)


def calculate_lightgbm_classifier_output_shapes(operator):
    '''
    This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
    appear in this operator's output list, we should further generate a map storing all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N, 1], A sequence of map

    Note that the second case is not allowed as long as ZipMap only produces dictionary.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, str) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
    else:
        operator.outputs[0].type = Int64TensorType(shape=[N])
    if len(class_labels) > 2:
        operator.outputs[1].type = FloatTensorType()
    else:
        operator.outputs[1].type = FloatTensorType(shape=[N, 1])


def calculate_lgbm_zipmap(operator):
    check_input_and_output_numbers(operator, output_count_range=2)


register_shape_calculator('LgbmClassifier', calculate_lightgbm_classifier_output_shapes)
register_shape_calculator('LgbmZipMap', calculate_lgbm_zipmap)
