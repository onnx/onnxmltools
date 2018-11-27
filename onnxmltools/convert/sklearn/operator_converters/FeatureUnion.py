# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._apply_operation import apply_abs, apply_cast, apply_mul, apply_reshape, apply_sub
from ...common._registration import get_converter, register_converter
from .._parse import sklearn_operator_name_map
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class Output:
    def __init__(self, full_name):
        self.full_name = full_name

class Oper:
    def __init__(self, model, inputs, op_type, input_full_names, output_full_name):
        self.raw_operator = model
        self.inputs = inputs
        self.input_full_names = input_full_names
        self.type = op_type
        self.outputs = [Output(output_full_name)]

def convert_sklearn_feature_union(scope, operator, container):
    op = operator.raw_operator
    transform_result_name = []
    supported_transformer_list = [PCA, TruncatedSVD]
    for name, transform in op.transformer_list:
        if type(transform) not in supported_transformer_list:
            raise NotImplementedError
        op_type = sklearn_operator_name_map[type(transform)]
        result_name = scope.get_unique_variable_name('result')
        this_operator = Oper(transform, operator.inputs, op_type, operator.input_full_names, result_name)
        get_converter(op_type)(scope, this_operator, container)
        transform_result_name.append(result_name)

    container.add_node('Concat', [s for s in transform_result_name],
                       operator.outputs[0].full_name, name=scope.get_unique_operator_name('Concat'), axis=1)


register_converter('SklearnFeatureUnion', convert_sklearn_feature_union)
