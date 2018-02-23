#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import utils
from . import TreeConverterCommon


class TreeEnsembleRegressorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'treeEnsembleRegressor')
            utils._check_has_attr(cm_node.treeEnsembleRegressor, 'treeEnsemble')
            utils._check_has_attr(cm_node.treeEnsembleRegressor.treeEnsemble, 'nodes')
            utils._check_has_attr(cm_node.treeEnsembleRegressor, 'postEvaluationTransform')
            utils._check_has_attr(cm_node.treeEnsembleRegressor.treeEnsemble, 'basePredictionValue')
        except AttributeError as e:
            raise RuntimeError('Missing type from CoreML node:' + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML TreeEnsembleRegressor to ONNX
        """
        return TreeConverterCommon.convert(context, cm_node, inputs, outputs, "target")


# Register the class for processing
register_converter("treeEnsembleRegressor", TreeEnsembleRegressorConverter)
