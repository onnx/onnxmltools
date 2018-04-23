# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import StringTensorType, Int64TensorType
from ...common._registration import register_converter


def convert_sklearn_label_encoder(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LabelEncoder'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['classes_strings'] = [str(c) for c in op.classes_]

    if isinstance(operator.inputs[0].type, Int64TensorType):
        attrs['default_int64'] = -1
    elif isinstance(operator.inputs[0].type, StringTensorType):
        attrs['default_string'] = '__unknown__'
    else:
        raise RuntimeError('Unsupported input type: %s' % type(operator.inputs[0].type))

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnLabelEncoder', convert_sklearn_label_encoder)
