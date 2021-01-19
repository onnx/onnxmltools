# SPDX-License-Identifier: Apache-2.0

import numbers
from ...common._apply_operation import apply_reshape
from ...common._registration import register_converter


def convert_tensor_to_probability_map(scope, operator, container):
    '''
    This converter tries to convert a special operator 'TensorToProbabilityMap' into a sequence of some ONNX operators.
    Those operators are used to create a dictionary in which keys are class labels and values are the associated
    probabilities. We assume that the elements in the given probability tensor are aligned with the class labels
    specified in the CoreML model.

    Notice that ONNX<1.2 doesn't support a CoreML classifier with a batch size larger than one because old ONNX ZipMap
    is not able to produce a sequence of dictionaries. This issue has been fixed in ONNX-1.2.
    '''
    attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = list(s.encode('utf-8') for s in model.stringClassLabels.vector)
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = list(int(i) for i in model.int64ClassLabels.vector)
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = list(s.encode('utf-8') for s in model.stringClassLabels.vector)
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = list(int(i) for i in model.int64ClassLabels.vector)
        else:
            raise ValueError('Unknown label type found')
    else:
        raise TypeError('Only neural network classifiers and pipeline classifiers are supported')

    input_shape = operator.inputs[0].type.shape
    if len(operator.inputs[0].type.shape) != 2:
        # Calculate the shape attribute of ONNX Reshape
        if input_shape[0] != 'None':
            N = input_shape[0]
        else:
            N = -1  # -1 means that this dimension is automatically determined in runtime and unknown in conversion time

        if all(isinstance(i, numbers.Integral) for i in input_shape[1:]):
            C = 1
            for i in input_shape[1:]:
                C *= int(i)
        else:
            C = -1  # -1 means that this dimension is automatically determined in runtime and unknown in conversion time

        # ZipMap in ONNX only accepts [C] and [N, C] inputs. In cases of [N, C, 1, 1], we reshape the probability tensor
        # into [N, C] before feeding it into ZipMap.
        buffer_name = scope.get_unique_variable_name('buffer')
        apply_reshape(scope, operator.inputs[0].full_name, buffer_name, container, desired_shape=[N, C])
    else:
        buffer_name = operator.inputs[0].full_name

    container.add_node('ZipMap', buffer_name, operator.outputs[0].full_name,
                       op_domain='ai.onnx.ml', **attrs)


register_converter('tensorToProbabilityMap', convert_tensor_to_probability_map)
