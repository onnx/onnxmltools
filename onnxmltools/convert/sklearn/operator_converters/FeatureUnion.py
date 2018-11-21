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

class Oper:
    def __init__(self, model, inputs, op_type):
        self.raw_operator = model
        self.inputs = inputs
        self.type = op_type

def convert_sklearn_feature_union(scope, operator, container):
    op = operator.raw_operator
    transform_result_name = [] 
    for name, transform in op.transformer_list:
        op_type = sklearn_operator_name_map[type(transform)] 
        this_operator = Oper(transform, operator.inputs, op_type)
        transform_result_name.append(get_converter(op_type)(scope, this_operator, container, flag=True))

    container.add_node('Concat', [s for s in transform_result_name],
                           operator.outputs[0].full_name, name=scope.get_unique_operator_name('Concat'), axis=1)


register_converter('SklearnFeatureUnion', convert_sklearn_feature_union)
