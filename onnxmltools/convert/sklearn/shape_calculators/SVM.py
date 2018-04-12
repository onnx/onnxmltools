# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import six, numbers
from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType, StringTensorType, DictionaryType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_svm_output_shapes(operator):
    op = operator.raw_operator

    N = operator.inputs[0].type.shape[0]

    if operator.type in ['SklearnSVC']:
        check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=[1, 2])

        if N != 1 and N != 'None':
            # In this case, output probability map should be a sequence of dictionaries, which is not implemented yet.
            raise RuntimeError('Currently batch size must be one')
        if len(operator.outputs) != 2:
            raise RuntimeError('Support vector classifier has two outputs')
        if all(isinstance(i, (six.string_types, six.text_type)) for i in op.classes_):
            operator.outputs[0].type = StringTensorType([1, 1])
            operator.outputs[1].type = DictionaryType(StringTensorType([1]), FloatTensorType([1]))
        elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in op.classes_):
            operator.outputs[0].type = Int64TensorType([1, 1])
            operator.outputs[1].type = DictionaryType(Int64TensorType([1]), FloatTensorType([1]))
        else:
            raise RuntimeError('Class labels should be either all strings or all integers')

    if operator.type in ['SklearnSVR']:
        check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=1)

        operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnSVC', calculate_sklearn_svm_output_shapes)
register_shape_calculator('SklearnSVR', calculate_sklearn_svm_output_shapes)
