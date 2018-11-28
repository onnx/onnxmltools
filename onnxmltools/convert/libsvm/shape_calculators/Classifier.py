# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType
from ...common.utils import check_input_and_output_numbers
from svm import C_SVC, NU_SVC


def calculate_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=2)

    N = operator.inputs[0].type.shape[0]
    if len(operator.outputs) != 2:
        raise RuntimeError("Expect only two outputs not {0}".format(len(operator.outputs)))
    svm_node =  operator.raw_operator
    if svm_node.is_probability_model():
        nc = N
    else:
        # libsvm produces n(n-1) raw scores.
        # onnxruntime aggregates the scores
        # but this behavior could be revisited.
        # st = svm_node.param.svm_type
        # if st == C_SVC or st == NU_SVC:
        #    nc = (N * (N-1)) // 2
        # else:
        #    nc = N
        nc = N
    svm_node.is_probability_model()
    operator.outputs[0].type = Int64TensorType([N, 1])
    operator.outputs[1].type = FloatTensorType([nc, 1])


register_shape_calculator('LibSvmSVC', calculate_classifier_output_shapes)
