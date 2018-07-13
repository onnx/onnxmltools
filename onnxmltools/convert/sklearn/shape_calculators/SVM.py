# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import six, numbers
from distutils.version import StrictVersion
from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType, SequenceType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_svm_output_shapes(operator):
    '''
    For SVM classifiers, allowed input/output patterns are
        1. [N, C] ---> [N], A sequence of map
    Note that the second case is not allowed as long as ZipMap only produces dictionary.

    For SVM regressors, allowed input/output patterns are
        1. [N, C] ---> [N]

    For both of SVC and SVR, the inputs should numerical tensor(s). For SVC with batch size 1, the first output is the
    label and the second output is a map used to store all class probabilities (For a key-value pair, the value is
    assigned to the class specified by the key). If batch size is larger than 1, we need to use a sequence of maps to
    denote class probabilities. Regarding SVR, we just produce a scalar for each example. If there are N examples, the
    output shape would be [N, 1].
    '''
    op = operator.raw_operator

    N = operator.inputs[0].type.shape[0]

    if operator.type in ['SklearnSVC']:
        check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=[1, 2])

        if all(isinstance(i, (six.string_types, six.text_type)) for i in op.classes_):
            operator.outputs[0].type = StringTensorType([N])
            if len(operator.outputs) == 2:
                if operator.targeted_onnx_version < StrictVersion('1.2'):
                    # Old ONNX ZipMap produces Map type
                    operator.outputs[1].type = \
                        DictionaryType(StringTensorType([1]), FloatTensorType([1]))
                else:
                    # New ONNX ZipMap produces Seq<Map> type
                    operator.outputs[1].type = \
                        SequenceType(DictionaryType(StringTensorType([]), FloatTensorType([])), N)
        elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in op.classes_):
            operator.outputs[0].type = Int64TensorType([N])
            if len(operator.outputs) == 2:
                if operator.targeted_onnx_version < StrictVersion('1.2'):
                    # Old ONNX ZipMap produces Map type
                    operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
                else:
                    # New ONNX ZipMap produces Seq<Map> type
                    operator.outputs[1].type = \
                        SequenceType(DictionaryType(Int64TensorType([]), FloatTensorType([])), N)
        else:
            raise RuntimeError('Class labels should be either all strings or all integers')

    if operator.type in ['SklearnSVR']:
        check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=1)

        operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnSVC', calculate_sklearn_svm_output_shapes)
register_shape_calculator('SklearnSVR', calculate_sklearn_svm_output_shapes)
