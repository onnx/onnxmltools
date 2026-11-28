# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter
from .SVC import _get_kernel_params, extract_support_vectors_as_dense_tensor


def convert_svm_regressor(scope, operator, container):
    params = operator.raw_operator.supportVectorRegressor

    kernel_enum = {
        "linearKernel": "LINEAR",
        "polyKernel": "POLY",
        "rbfKernel": "RBF",
        "sigmoidKernel": "SIGMOID",
        "precomputedKernel": "PRECOMPUTED",
    }
    kernel = params.kernel
    kernel_val = kernel.WhichOneof("kernel")
    svr_kernel = kernel_enum[kernel_val]
    svr_kernel_params = _get_kernel_params(kernel, kernel_val)

    n_supports, support_vectors = extract_support_vectors_as_dense_tensor(
        operator.raw_operator.supportVectorRegressor
    )

    svr_coefficients = params.coefficients.alpha
    if isinstance(params.rho, list):
        svr_rho = [-x for x in params.rho]
    else:
        svr_rho = [-params.rho]

    op_type = "SVMRegressor"
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {"name": op_name}
    attrs["kernel_type"] = svr_kernel
    attrs["kernel_params"] = svr_kernel_params
    attrs["support_vectors"] = support_vectors
    attrs["n_supports"] = n_supports
    attrs["coefficients"] = svr_coefficients
    attrs["rho"] = svr_rho

    container.add_node(
        op_type,
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        **attrs,
    )


register_converter("supportVectorRegressor", convert_svm_regressor)
