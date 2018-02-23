#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...proto import onnx_proto
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util


class ArrayFeatureExtractorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'arrayFeatureExtractor')
        except AttributeError as e:
            raise RuntimeError('Missing type from coreml node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        nb = NodeBuilder(context, 'ArrayFeatureExtractor')
        nb.extend_inputs(inputs)
        target_index = cm_node.arrayFeatureExtractor.extractIndex
        index_tensor = model_util.make_tensor('TargetIndex', onnx_proto.TensorProto.INT64, [len(target_index)], target_index)
        nb.add_initializer(index_tensor)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
register_converter("arrayFeatureExtractor", ArrayFeatureExtractorConverter)
