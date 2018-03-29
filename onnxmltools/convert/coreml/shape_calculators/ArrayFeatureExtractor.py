# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ...common._data_types import TensorType
from ...common._registration import register_shape_calculator


def calculate_array_feature_extractor_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Array feature extractor has only one input and one output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

    N = operator.inputs[0].type.shape[0]
    extracted_feature_number = len(operator.raw_operator.arrayFeatureExtractor.extractIndex)

    # Save doc_string before over-writing by us
    doc_string = operator.outputs[0].type.doc_string
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, extracted_feature_number]
    # Assign correct doc_string to the output
    operator.outputs[0].type.doc_string = doc_string


register_shape_calculator('arrayFeatureExtractor', calculate_array_feature_extractor_output_shapes)
