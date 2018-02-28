#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import NodeBuilder 
from ..common import utils


class NormalizerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'normalizer')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML Normalizer to ONNX
        """
        norms = ['MAX', 'L1', 'L2']
        nb = NodeBuilder(context, 'Normalizer', op_domain='ai.onnx.ml')
        if cm_node.normalizer.normType in range(3):
            nb.add_attribute('norm', norms[cm_node.normalizer.normType])
        else:
            raise RuntimeError('Invalid norm type: ' + cm_node.normalizer.normType)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
register_converter('normalizer', NormalizerConverter)
