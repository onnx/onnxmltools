#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from sklearn import linear_model
from sklearn import svm
from ...proto import onnx_proto
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
import numpy as np


class GLMRegressorConverter:
    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'coef_')
            utils._check_has_attr(sk_node, 'intercept_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        coefficients = sk_node.coef_.tolist()
        intercepts = sk_node.intercept_
        if not isinstance(intercepts, np.ndarray):
            intercepts = [intercepts]

        nb = NodeBuilder(context, 'LinearRegressor', op_domain='ai.onnx.ml')
        nb.add_attribute('coefficients', coefficients)
        nb.add_attribute('intercepts', intercepts)

        nb.extend_inputs(inputs)
        try:
            output_type = inputs[0].type.tensor_type.elem_type
        except AttributeError as e:
            raise ValueError('Invalid or missing input type for GLMRegressor.')
        if output_type == onnx_proto.TensorProto.STRING:
            raise ValueError('Invalid or missing input type for GLMRegressor.')
        output_dim = None
        try:
            if len(inputs[0].type.tensor_type.shape.dim) > 0:
                output_dim = [1, len(intercepts)]
        except AttributeError as e:
            raise ValueError('Invalid or missing input dimension for GLMRegressor.')
        nb.add_output(model_util.make_tensor_value_info(nb.name, output_type, output_dim))

        return nb.make_node()


# Register the class for processing
register_converter(svm.LinearSVR, GLMRegressorConverter)
register_converter(linear_model.LinearRegression, GLMRegressorConverter)
register_converter(linear_model.Ridge, GLMRegressorConverter)
register_converter(linear_model.SGDRegressor, GLMRegressorConverter)
