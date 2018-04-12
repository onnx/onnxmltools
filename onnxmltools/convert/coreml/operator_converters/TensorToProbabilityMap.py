# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter


def convert_tensor_to_probability_map(scope, operator, container):
    '''
    This converter tries to convert a special operator 'TensorToProbabilityMap' into a sequence of some ONNX operators.
    Those operators are used to create a dictionary in which keys are class labels and values are the associated
    probabilities. We assume that the elements in the given probability tensor are aligned with the class labels
    specified in the CoreML model.

    Notice that we currently doesn't support a CoreML classifier with a batch size larger than one because ONNX's ZipMap
    is not able to produce a batch of dictionaries.
    '''
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
