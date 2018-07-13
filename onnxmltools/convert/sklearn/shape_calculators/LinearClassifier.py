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
    if all(isinstance(i, (six.string_types, six.text_type)) for i in class_labels):
        operator.outputs[0].type = StringTensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.targeted_onnx_version < StrictVersion('1.2'):
                operator.outputs[1].type = DictionaryType(StringTensorType([1]), FloatTensorType([1]))
            else:
                operator.outputs[1].type = SequenceType(DictionaryType(StringTensorType([]), FloatTensorType([])), N)
        else:
            # For binary classifier, we produce the probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[N, 1])
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
        operator.outputs[0].type = Int64TensorType(shape=[N])
        if len(class_labels) > 2 or operator.type != 'SklearnLinearSVC':
            # For multi-class classifier, we produce a map for encoding the probabilities of all classes
            if operator.targeted_onnx_version < StrictVersion('1.2'):
                operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
            else:
                operator.outputs[1].type = SequenceType(DictionaryType(Int64TensorType([]), FloatTensorType([])), N)
        else:
            # For binary classifier, we produce the probability of the positive class
            operator.outputs[1].type = FloatTensorType(shape=[N, 1])
    else:
        raise ValueError('Unsupported or mixed label types')


register_shape_calculator('SklearnLinearClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnLinearSVC', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnDecisionTreeClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreesClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier', calculate_sklearn_linear_classifier_output_shapes)
register_shape_calculator('LgbmClassifier', calculate_sklearn_linear_classifier_output_shapes)
