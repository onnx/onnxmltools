# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .._data_types import FloatTensorType
from ..registration import register_converter


def convert_tensor_to_probability_map(scope, operator, container):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Too many input or output variables')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Only float tensor is supported')

    attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = list(s.encode('ascii') for s in model.stringClassLabels.vector)
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = list(int(i) for i in model.int64ClassLabels.vector)
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = list(s.encode('ascii') for s in model.stringClassLabels.vector)
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = list(int(i) for i in model.int64ClassLabels.vector)
        else:
            raise ValueError('Unknown label type found')
    else:
        raise TypeError('Only neural network classifiers and pipeline classifiers are supported')

    container.add_node('ZipMap', [operator.inputs[0].full_name], [operator.outputs[0].full_name],
                       op_domain='ai.onnx.ml', **attrs)


register_converter('tensorToProbabilityMap', convert_tensor_to_probability_map)
