# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType, Int64Type, StringTensorType, StringType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculte_tensor_to_label_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    Note that N must be 1 currently because TensorToProbability doesn't support batch size larger than 1.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    N = operator.inputs[0].type.shape[0]
    if operator.targeted_onnx_version < StrictVersion('1.2'):
        output_shape = [1, 1]
    else:
        output_shape = [N, 1]

    if type(operator.outputs[0].type) in [Int64Type, Int64TensorType]:
        operator.outputs[0].type = Int64TensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)
    elif type(operator.outputs[0].type) in [StringType, StringTensorType]:
        operator.outputs[0].type = StringTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)
    else:
        raise ValueError('Unsupported label type')


register_shape_calculator('tensorToLabel', calculte_tensor_to_label_output_shapes)
