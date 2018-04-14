# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType, Int64Type, StringTensorType, StringType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculte_tensor_to_label_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [1, C] ---> [1]
        2. [N, C] ---> [N , 1]

    Note that N must be 1 currently because TensorToProbability doesn't support batch size larger than 1.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    N = operator.inputs[0].type.shape[0]
    if type(operator.outputs[0].type) == Int64Type:
        operator.outputs[0].type = Int64TensorType([1], doc_string=operator.outputs[0].type.doc_string)
        # Due to the limitation of ZipMap, we are not able to produce label and class probability map for batch size
        # greater than 1. It leads to that although the following code is semantically correct, we cannot use it.
        # if N == 1:
        #    operator.outputs[0].type = Int64Type()
        # else:
        #    operator.outputs[0].type = Int64TensorType([N, 1])
    elif type(operator.outputs[0].type) == StringType:
        operator.outputs[0].type = StringTensorType([1], doc_string=operator.outputs[0].type.doc_string)
        # Due to the limitation of ZipMap, we are not able to produce label and class probability map for batch size
        # greater than 1. It leads to that although the following code is semantically correct, we cannot use it.
        # if N == 1:
        #    operator.outputs[0].type = StringTensorType([N, 1])
        # else:
        #    operator.outputs[0].type = StringType()
    else:
        raise ValueError('Unsupported label type')


register_shape_calculator('tensorToLabel', calculte_tensor_to_label_output_shapes)
