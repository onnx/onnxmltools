# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
from ...proto import onnx_proto


class FeatureVectorizerConverter:

    @staticmethod
    def validate(coreml_node):
        try:
            utils._check_has_attr(coreml_node, 'featureVectorizer')
            utils._check_has_attr(coreml_node.featureVectorizer, 'inputList')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML FeatureVectorizer to ONNX
        """
        input_col = []
        input_dim = []
        nodes = []

        # Iterate through the inputs in the coreml description to determine if any are
        # an integer input. If so, add a scaler operation to convert to float
        for idx, input_val in enumerate(cm_node.featureVectorizer.inputList):
            input_desc = cm_node.description.input[idx]
            if input_desc.type.WhichOneof('Type') == 'int64Type':
                # create a temporary input
                onnx_input = model_util.make_tensor_value_info(inputs[idx],
                                                               onnx_proto.TensorProto.INT64,
                                                               [1, input_val.inputDimensions])
                node = model_util.create_scaler(onnx_input, input_desc.name, 1.0, 0.0, context)
                input_col.append(node.outputs[0].name)
                input_dim.append(node.outputs[0].type.tensor_type.shape.dim[-1].dim_value)
                nodes.append(node)
            else:
                input_col.append(inputs[idx])
                input_dim.append(int(input_val.inputDimensions))

        nb = NodeBuilder(context, 'FeatureVectorizer')
        nb.add_attribute('inputdimensions', input_dim)

        nb.extend_inputs(input_col)
        nb.extend_outputs(outputs)
        nodes.append(nb.make_node())
        return nodes


# Register the class for processing
register_converter("featureVectorizer", FeatureVectorizerConverter)

