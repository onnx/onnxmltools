# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils

class DictVectorizerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'dictVectorizer')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML DictVectorizer to ONNX
        """

        nb = NodeBuilder(context, 'DictVectorizer', op_domain='ai.onnx.ml')
        if cm_node.dictVectorizer.HasField('stringToIndex'):
            nb.add_attribute('string_vocabulary', cm_node.dictVectorizer.stringToIndex.vector)
        else:
            nb.add_attribute('int64_vocabulary', cm_node.dictVectorizer.int64ToIndex.vector)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        return nb.make_node()


# Register the class for processing
register_converter('dictVectorizer', DictVectorizerConverter)
