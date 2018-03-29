# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import Int64TensorType, Int64Type, StringTensorType, StringType, TensorType
from ...common._registration import register_shape_calculator


def calculte_tensor_to_label_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-label operator has only one input and output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

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
