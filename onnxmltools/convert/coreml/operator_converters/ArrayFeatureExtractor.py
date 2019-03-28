# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter


def convert_array_feature_extractor(scope, operator, container):
    op_type = 'ArrayFeatureExtractor'
    attrs = {'name': operator.full_name}

    target_indexes = operator.raw_operator.arrayFeatureExtractor.extractIndex
    index_buffer_name = scope.get_unique_variable_name('target_indexes')
    container.add_initializer(index_buffer_name, onnx_proto.TensorProto.INT64, [len(target_indexes)], target_indexes)

    inputs = [operator.inputs[0].full_name, index_buffer_name]
    outputs = [operator.outputs[0].full_name]

    container.add_node(op_type, inputs, outputs, op_domain='ai.onnx.ml', **attrs)


register_converter('arrayFeatureExtractor', convert_array_feature_extractor)
