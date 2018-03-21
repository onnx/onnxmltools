# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ..registration import register_shape_calculator


def calculate_identity_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Identity layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    doc_string = output.type.doc_string
    output.type = copy.deepcopy(input.type)
    output.type.doc_string = doc_string


register_shape_calculator('identity', calculate_identity_output_shapes)
register_shape_calculator('imputer', calculate_identity_output_shapes)
register_shape_calculator('scaler', calculate_identity_output_shapes)
register_shape_calculator('normalizer', calculate_identity_output_shapes)

