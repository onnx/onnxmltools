# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import FloatTensorType
from ...common._registration import register_shape_calculator


def calculate_sklearn_dict_vectorizer_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Only one input and one output are allowed')
    C = len(operator.raw_operator.feature_names_)

    operator.outputs[0].type = FloatTensorType([1, C])


register_shape_calculator('SklearnDictVectorizer', calculate_sklearn_dict_vectorizer_output_shapes)
