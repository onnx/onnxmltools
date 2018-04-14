# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_array_feature_extractor_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, C']

    C' is the number of extracted features.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType, StringTensorType])

    N = operator.inputs[0].type.shape[0]
    extracted_feature_number = len(operator.raw_operator.arrayFeatureExtractor.extractIndex)

    # Save doc_string before over-writing by us
    doc_string = operator.outputs[0].type.doc_string
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, extracted_feature_number]
    # Assign correct doc_string to the output
    operator.outputs[0].type.doc_string = doc_string


register_shape_calculator('arrayFeatureExtractor', calculate_array_feature_extractor_output_shapes)
