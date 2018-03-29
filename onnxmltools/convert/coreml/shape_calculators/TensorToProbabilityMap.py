# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import Int64Type, StringType, DictionaryType, FloatTensorType
from ...common._registration import register_shape_calculator


def calculate_tensor_to_probability_map_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-probability operator has only one input and output')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        class_label_type = operator.raw_operator.neuralNetworkClassifier.WhichOneof('ClassLabels')
    else:
        raise TypeError('%s has no class label' % model_type)

    N = operator.inputs[0].type.shape[0]
    doc_string = operator.outputs[0].type.doc_string
    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = DictionaryType(StringType(), FloatTensorType([1]), doc_string=doc_string)
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(StringType(), FloatType()), N)
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = DictionaryType(Int64Type(), FloatTensorType([1]), doc_string=doc_string)
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(Int64Type(), FloatType()), N)
    else:
        raise ValueError('Unsupported label type')


register_shape_calculator('tensorToProbabilityMap', calculate_tensor_to_probability_map_output_shapes)
