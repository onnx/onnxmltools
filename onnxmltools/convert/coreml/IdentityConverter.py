#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import NodeBuilder


class IdentityConverter:

    @staticmethod
    def validate(cm_node):
        pass

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML Identity to ONNX
        """
        nb = NodeBuilder(context, 'Identity')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
register_converter('identity', IdentityConverter)
