#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import utils
from ..common import NodeBuilder


class ScalerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'scaler')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """Converts a CoreML Scaler to ONNX"""
        scale = [x for x in cm_node.scaler.scaleValue]
        offset = [-x for x in cm_node.scaler.shiftValue]

        nb = NodeBuilder(context, 'Scaler', op_domain='ai.onnx.ml')
        nb.add_attribute('scale', scale)
        nb.add_attribute('offset', offset)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        return nb.make_node()


# Register the class for processing
register_converter("scaler", ScalerConverter)
