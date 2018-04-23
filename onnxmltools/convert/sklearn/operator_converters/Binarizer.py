# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter
from .common import concatenate_variables


def convert_sklearn_binarizer(scope, operator, container):
    feature_name = concatenate_variables(scope, operator.inputs, container)

    op_type = 'Binarizer'
    attrs = {'name': scope.get_unique_operator_name(op_type), 'threshold': float(operator.raw_operator.threshold)}
    container.add_node(op_type, feature_name, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnBinarizer', convert_sklearn_binarizer)
