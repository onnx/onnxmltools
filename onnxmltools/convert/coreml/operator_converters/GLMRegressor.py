# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ...common._registration import register_converter


def convert_glm_regressor(scope, operator, container):
    from coremltools.proto.GLMRegressor_pb2 import GLMRegressor

    op_type = 'LinearRegressor'
    glm = operator.raw_operator.glmRegressor
    attrs = {'name': operator.full_name}

    transform_table = {GLMRegressor.NoTransform: 'NONE', GLMRegressor.Logit: 'LOGISTIC', GLMRegressor.Probit: 'PROBIT'}
    if glm.postEvaluationTransform in transform_table:
        attrs['post_transform'] = transform_table[glm.postEvaluationTransform]
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

    attrs['targets'] = dim_target
    attrs['coefficients'] = matrix_w.flatten()
    attrs['intercepts'] = glm.offset

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('glmRegressor', convert_glm_regressor)
