#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from ..common import NodeBuilder
from ..common import register_converter
from ..common import utils
from .SupportVectorClassifierConverter import extract_support_vectors_as_dense_tensor


class SupportVectorRegressorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'supportVectorRegressor')
            utils._check_has_attr(cm_node.supportVectorRegressor, 'kernel')
            utils._check_has_attr(cm_node.supportVectorRegressor, 'coefficients')
            utils._check_has_attr(cm_node.supportVectorRegressor.coefficients, 'alpha')
            utils._check_has_attr(cm_node.supportVectorRegressor, 'rho')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """Converts a CoreML SVR to ONNX"""
        kernel_enum = {'linearKernel': 'LINEAR', 'polyKernel': 'POLY',
                       'rbfKernel': 'RBF', 'sigmoidKernel': 'SIGMOID', 'precomputedKernel': 'PRECOMPUTED'}
        kernel = cm_node.supportVectorRegressor.kernel
        kernel_val = kernel.WhichOneof('kernel')
        svr_kernel = kernel_enum[kernel_val]

        if kernel_val == 'rbfKernel':
            svr_kernel_params = [kernel.rbfKernel.gamma, 0.0, 0.0]
        elif kernel_val == 'polyKernel':
            svr_kernel_params = [kernel.polyKernel.gamma, kernel.polyKernel.coef0, kernel.polyKernel.degree]
        elif kernel_val == 'sigmoidKernel':
            svr_kernel_params = [kernel.sigmoidKernel.gamma, kernel.sigmoidKernel.coef0, 0.0]
        elif kernel_val == 'linearKernel':
            svr_kernel_params = [0.0, 0.0, 0.0]
        else:
            raise ValueError('Unsupported kernel type: %s' % kernel_val)

        n_supports, support_vectors = extract_support_vectors_as_dense_tensor(cm_node.supportVectorRegressor)

        svr_coefficients = cm_node.supportVectorRegressor.coefficients.alpha
        if isinstance(cm_node.supportVectorRegressor.rho, list):
            svr_rho = [-x for x in cm_node.supportVectorRegressor.rho]
        else:
            svr_rho = [-cm_node.supportVectorRegressor.rho]

        nb = NodeBuilder(context, 'SVMRegressor', op_domain='ai.onnx.ml')
        nb.add_attribute('kernel_type', svr_kernel)
        nb.add_attribute('kernel_params', svr_kernel_params)
        nb.add_attribute('support_vectors', support_vectors)
        nb.add_attribute('n_supports', n_supports)
        nb.add_attribute('coefficients', svr_coefficients)
        nb.add_attribute('rho', svr_rho)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
register_converter("supportVectorRegressor", SupportVectorRegressorConverter)
