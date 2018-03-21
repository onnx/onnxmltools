# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import helper
from ....proto import onnx_proto
from ..registration import register_converter


def convert_tensor_to_label(scope, operator, container):
    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = list(s.encode('ascii') for s in model.stringClassLabels.vector)
            label_type = onnx_proto.TensorProto.STRING
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            labels = list(int(i) for i in model.int64ClassLabels.vector)
            label_type = onnx_proto.TensorProto.INT64
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = list(s.encode('ascii') for s in model.pipelineClassifier.stringClassLabels.vector)
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
    label_loader_attrs = {'name': label_loader_name}
    label_buffer_name = scope.get_unique_variable_name('ClassLabels')
    label_loader_attrs['value'] = helper.make_tensor(label_buffer_name, label_type, [len(labels)], labels)
    container.add_node('Constant', [], [label_buffer_name], **label_loader_attrs)

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
