# SPDX-License-Identifier: Apache-2.0

from ....proto import helper
from ....proto import onnx_proto
from ...common._registration import register_converter
from ...common._apply_operation import apply_constant

def convert_tensor_to_label(scope, operator, container):
    '''
    This converter tries to convert a dummy operator 'TensorToLabel' into a sequence of some ONNX operators. Those
    operators are used to extract the label with the highest probability for doing a prediction. We assume that the
    elements in the given probability tensor are aligned with the class labels specified in the CoreML model. That is,
    if you have a class label vector ['a', 'b'] in our CoreML classifier, the first (and the only) input of this
    operator should be [probability_of_class_a, probability_of_class_b].

    Assume that we have C classes with batch size N (N must be 1. If not, the output class probabilities need to be
    encoded as a sequence of dictionary, which is not allowed in ONNX). The ONNX computation graph of this operator may
    look like

                Probability tensor [1, C] (the variable defined at operator.inputs[0])
                        |
                        v
                      ArgMax                                        LoadConstant (its attribute is extracted from
                        |                                                |        operator.raw_operator, which is a
                        |                                                |        CoreML classifier)
                        |                                                |
                        v                                                |
                   best index [1]                                        |
                        |                                                |
                        v                                                v
               ArrayFeatureExtractor  <-------------------- a 1-D tensor of class labels [C]
                        |
                        v
                  predicted label [1]
    '''
    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = list(s.encode('utf-8') for s in model.stringClassLabels.vector)
            label_type = onnx_proto.TensorProto.STRING
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            labels = list(int(i) for i in model.int64ClassLabels.vector)
            label_type = onnx_proto.TensorProto.INT64
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = list(s.encode('utf-8') for s in model.pipelineClassifier.stringClassLabels.vector)
            label_type = onnx_proto.TensorProto.STRING
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            labels = list(int(i) for i in model.int64ClassLabels.vector)
            label_type = onnx_proto.TensorProto.INT64
        else:
            raise ValueError('Unknown label type found')
    else:
        raise ValueError('Only neural network classifiers and pipeline classifiers are supported')

    # Use a Constant operator to load and output all labels as a tensor
    label_loader_name = scope.get_unique_operator_name('LabelLoader')
    label_buffer_name = scope.get_unique_variable_name('ClassLabels')
    label_loader_value = helper.make_tensor(label_buffer_name, label_type, [len(labels)], labels)
    apply_constant(scope, [label_buffer_name], container,
                    operator_name=label_loader_name, value=label_loader_value)

    # Extract most possible label index
    label_id_extractor_name = scope.get_unique_operator_name('LabelIndexExtractor')
    label_id_extractor_attrs = {'name': label_id_extractor_name}
    label_id_extractor_attrs['axis'] = 1
    label_id_extractor_attrs['keepdims'] = 1
    extracted_id_name = scope.get_unique_variable_name('LabelId')
    container.add_node('ArgMax', [operator.inputs[0].full_name], [extracted_id_name], **label_id_extractor_attrs)

    # Pick up the label indicated by the selected ID
    label_selector_name = scope.get_unique_operator_name('LabelSelector')
    label_selector_attrs = {'name': label_selector_name}
    container.add_node('ArrayFeatureExtractor', [label_buffer_name, extracted_id_name], [operator.outputs[0].full_name],
                       op_domain='ai.onnx.ml', **label_selector_attrs)


register_converter('tensorToLabel', convert_tensor_to_label)
