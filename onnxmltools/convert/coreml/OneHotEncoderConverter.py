#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import utils
from ..common import NodeBuilder


class OneHotEncoderConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'oneHotEncoder')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML OneHotEncoder to ONNX
        """
        nb = NodeBuilder(context, 'OneHotEncoder')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        if cm_node.oneHotEncoder.HasField('int64Categories'):
            nb.add_attribute('cats_int64s', cm_node.oneHotEncoder.int64Categories.vector)
        if cm_node.oneHotEncoder.HasField('stringCategories'):
            nb.add_attribute('cats_strings', cm_node.oneHotEncoder.stringCategories.vector)

        return nb.make_node()


# Register the class for processing
register_converter("oneHotEncoder", OneHotEncoderConverter)
