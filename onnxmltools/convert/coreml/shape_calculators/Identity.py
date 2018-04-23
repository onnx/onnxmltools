# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ...common._registration import register_shape_calculator
from ...common.utils import check_input_and_output_numbers


def calculate_identity_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    input = operator.inputs[0]
    output = operator.outputs[0]

    doc_string = output.type.doc_string
    output.type = copy.deepcopy(input.type)
    output.type.doc_string = doc_string


register_shape_calculator('identity', calculate_identity_output_shapes)
register_shape_calculator('imputer', calculate_identity_output_shapes)
register_shape_calculator('scaler', calculate_identity_output_shapes)
register_shape_calculator('normalizer', calculate_identity_output_shapes)

