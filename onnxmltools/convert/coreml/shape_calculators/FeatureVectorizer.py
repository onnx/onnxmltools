# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import FloatTensorType, FloatType, Int64TensorType, Int64Type
from ...common._registration import register_shape_calculator


def calculate_feature_vectorizer_output_shapes(operator):
    if len(operator.outputs) != 1:
        raise RuntimeError('Feature vectorizer operator has only one output')
    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be float or integer tensor(s)')
    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be 2-D tensor(s)')

    # Find the first batch size which is not unknown
    N = 'None'
    for variable in operator.inputs:
        if variable.type.shape[0] != 'None':
            N = variable.type.shape[0]
            break
    for variable in operator.inputs:
        if variable.type.shape[0] not in ['None', N]:
            raise RuntimeError('The batch dimensions should be the same to all input tensors.')

    C = sum(info.inputDimensions for info in operator.raw_operator.featureVectorizer.inputList)

    if isinstance(operator.inputs[0].type, (FloatTensorType, FloatType)):
        doc_string = operator.outputs[0].type.doc_string
        operator.outputs[0].type = FloatTensorType([N, C], doc_string=doc_string)
    elif isinstance(operator.inputs[0].type, (Int64TensorType, Int64Type)):
        doc_string = operator.outputs[0].type.doc_string
        operator.outputs[0].type = Int64TensorType([N, C], doc_string=doc_string)
    else:
        raise ValueError('Unsupported input type: %s' % type(operator.inputs[0].type))


register_shape_calculator('featureVectorizer', calculate_feature_vectorizer_output_shapes)
