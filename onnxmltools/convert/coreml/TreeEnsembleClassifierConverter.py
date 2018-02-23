#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import utils
from . import TreeConverterCommon


class TreeEnsembleClassifierConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'treeEnsembleClassifier')
            utils._check_has_attr(cm_node.treeEnsembleClassifier, 'treeEnsemble')
            utils._check_has_attr(cm_node.treeEnsembleClassifier.treeEnsemble, 'nodes')
            utils._check_has_attr(cm_node.treeEnsembleClassifier, 'postEvaluationTransform')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML TreeEnsembleClassifier to ONNX
        """
        return TreeConverterCommon.convert(context, cm_node, inputs, outputs, "class")


# Register the class for processing
register_converter("treeEnsembleClassifier", TreeEnsembleClassifierConverter)
