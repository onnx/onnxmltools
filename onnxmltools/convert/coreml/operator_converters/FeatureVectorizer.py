# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import Int64Type, Int64TensorType
from ...common._registration import register_converter


def convert_feature_vectorizer(scope, operator, container):
    op_type = 'FeatureVectorizer'
    attrs = {'name': operator.full_name}

    inputs = []
    input_dims = []
    for variable in operator.inputs:
        if type(variable.type) in [Int64TensorType, Int64Type]:
            # We use scaler to convert integers into floats because output is a single tensor and all tensor elements
            # should be in the same type.
            scaler_name = scope.get_unique_operator_name('Scaler')
            scaled_name = scope.get_unique_variable_name(variable.full_name + '_scaled')
            scaler_attrs = {'name': scaler_name, 'scale': [1.], 'offset': [0.]}
            container.add_node('Scaler', [variable.full_name], [scaled_name], op_domain='ai.onnx.ml', **scaler_attrs)
            inputs.append(scaled_name)
        else:
            inputs.append(variable.full_name)
        # We assume feature vectorizer always combines inputs with shapes [1, C] or [C]
        input_dims.append(variable.type.shape[1])
    attrs['inputdimensions'] = input_dims

    container.add_node(op_type, inputs, [operator.outputs[0].full_name], op_domain='ai.onnx.ml', **attrs)


register_converter('featureVectorizer', convert_feature_vectorizer)
