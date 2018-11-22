# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import Int64TensorType, StringTensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_feature_union(operator):
    check_input_and_output_numbers(operator, output_count_range=1)


register_shape_calculator('SklearnFeatureUnion', calculate_sklearn_feature_union)
