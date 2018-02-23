#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.FeatureTypes_pb2 import ImageFeatureType


class ImageScalerPreprocessorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'scaler')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network preprocessing')

    @staticmethod
    def convert(context, cm_node, input, output):
        extend_inputs_from_2d_to_4d(context, input)

        nb = NodeBuilder(context, 'ImageScaler')
        nb.add_input(input)
        nb.add_output(output)

        params = cm_node.scaler
        # The scale parameter in CoreML is always a scalar. We just copy it and let ONNX scaler to broadcast it to all
        # channels.
        nb.add_attribute('scale', params.channelScale)

        # Deduce the feature to be processed. If no feature name is specified in CoreML's ImageScaler, we use the first
        # input of the neural network.
        network_inputs = context.data.description.input
        source_name = cm_node.featureName if cm_node.featureName != '' else network_inputs[0].name

        # Arrange biases for different channels according to the specified color space
        target_type = next(i.type.imageType for i in network_inputs if i.name == source_name)
        if target_type.colorSpace == ImageFeatureType.GRAYSCALE:
            bias = [params.grayBias]
        elif target_type.colorSpace == ImageFeatureType.RGB:
            bias = [params.redBias, params.greenBias, params.blueBias]
        elif target_type.colorSpace == ImageFeatureType.BGR:
            bias = [params.blueBias, params.greenBias, params.redBias]
        else:
            raise ValueError('Unsupported color space')
        nb.add_attribute('bias', bias)

        return nb.make_node()


registration.register_nn_converter('scaler', ImageScalerPreprocessorConverter)
