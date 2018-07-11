# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import lightgbm
import numpy as np
import six, numbers
from distutils.version import StrictVersion
from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_linear_classifier_output_shapes(operator):
    '''
    This operator maps an input feature vector into a scalar label if the number of outputs is one. If two outputs
    appear in this operator's output list, we should further generate a map storing all classes' probabilities.

    Allowed input/output patterns are
        1. [N, C] ---> [N], [N, E] (binary-class)
        2. [N, 1], ..., [N, 1] ---> [N], [N, E] (binary-class)
        3. [N, C] ---> [N], A sequence of map (multi-class)
        4. [N, 1], ..., [N, 1] ---> [N], A sequence of map (multi-class)

    Note that the second case is not allowed as long as ZipMap only produces dictionary.
    '''
    check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    if len(operator.inputs) > 1 and all((i.type.shape[1] != 1) for i in operator.inputs):
        raise RuntimeError('Multiple inputs must be a [N, 1]-tensors')

    N = operator.inputs[0].type.shape[0]

    class_labels = operator.raw_operator.classes_
    if all(isinstance(i, np.ndarray) for i in class_labels):
        class_labels = np.concatenate(class_labels)
    if all(isinstance(i, (six.string_types, six.text_type)) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N, 1])
        operator.outputs[1].type = FloatTensorType(shape=[N, len(class_labels)])
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N, 1])
        operator.outputs[1].type = FloatTensorType(shape=[N, len(class_labels)])
    else:
        raise ValueError('Unsupported or mixed label types')


register_shape_calculator('SklearnLinearClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnLinearSVC', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnDecisionTreeClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreesClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('LgbmClassifier', calculate_sklearn_linear_classifier_output_shapes)
