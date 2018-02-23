# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from coremltools.proto.GLMRegressor_pb2 import GLMRegressor


class GLMRegressorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'glmRegressor')
            utils._check_has_attr(cm_node.glmRegressor, 'weights')
            utils._check_has_attr(cm_node.glmRegressor, 'offset')
            utils._check_has_attr(cm_node.glmRegressor,
                                  'postEvaluationTransform')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        glm = cm_node.glmRegressor
        nb = NodeBuilder(context, 'LinearRegressor')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        transform_table = {GLMRegressor.NoTransform: 'NONE',
                           GLMRegressor.Logit: 'LOGISTIC',
                           GLMRegressor.Probit: 'PROBIT'}
        if glm.postEvaluationTransform in transform_table:
            nb.add_attribute('post_transform', transform_table[glm.postEvaluationTransform])
        else:
            raise ValueError('Unsupported post-transformation: {}'.format(glm.postEvaluationTransform))

        # Determine the dimensionality of the model weights. Conceptually,
        # the shape of the weight matrix in CoreML is E-by-F, where E and F
        # respectively denote the number of target variables and the number
        # of used features. Note that in ONNX, the shape is F-by-E.
        dim_target = len(glm.weights)
        dim_feature = len(glm.weights[0].value)

        matrix_w = np.ndarray(shape=(dim_feature, dim_target))
        for i, w in enumerate(glm.weights):
            matrix_w[:, i] = w.value

        nb.add_attribute('targets', dim_target)
        nb.add_attribute('coefficients', matrix_w.flatten())
        nb.add_attribute('intercepts', glm.offset)

        return nb.make_node()


# Register the class for processing
register_converter("glmRegressor", GLMRegressorConverter)
