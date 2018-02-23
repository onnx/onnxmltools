#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d


class CropLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'crop')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        # CoreML can have another input as the reference to crop the first input,
        # but such a function is not supported in ONNX.
        if len(inputs) > 1:
            raise ValueError('ONNX only supports cropping with a single input')

        nb = NodeBuilder(context, 'Crop')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        border = cm_node.crop.cropAmounts.borderAmounts
        left = border[1].startEdgeSize
        top = border[0].startEdgeSize
        right = border[1].endEdgeSize
        bottom = border[0].endEdgeSize
        nb.add_attribute('border', [left, top, right, bottom])

        return nb.make_node()


registration.register_nn_converter('crop', CropLayerConverter)
