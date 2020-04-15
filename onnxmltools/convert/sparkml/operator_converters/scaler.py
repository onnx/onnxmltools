# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy

import numpy
from pyspark.ml.feature import MaxAbsScalerModel, MinMaxScalerModel, StandardScalerModel

from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_scaler(scope, operator, container):
    op = operator.raw_operator
    input_name = operator.inputs[0].full_name

    op_type = 'Scaler'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    if isinstance(op, StandardScalerModel):
        C = operator.inputs[0].type.shape[1]
        attrs['offset'] = op.mean if op.getOrDefault("withMean") else [0.0] * C
        attrs['scale'] = [1.0 / x for x in op.std] if op.getOrDefault("withStd") else [1.0] * C
    elif isinstance(op, MinMaxScalerModel):
        epsilon = 1.0e-8  # to avoid dividing by zero
        attrs['offset'] = [x for x in op.originalMin]
        max_min = [a - b for a, b in zip(op.originalMax, op.originalMin)]
        attrs['scale'] = [1.0 / (x + epsilon) for x in max_min]
    elif isinstance(op, MaxAbsScalerModel):
        C = operator.inputs[0].type.shape[1]
        attrs['offset'] = [0.] * C
        attrs['scale'] = [1.0 / x for x in op.maxAbs]
    else:
        raise ValueError('Unsupported Scaler: %s' % type(op))

    # ONNX does not convert arrays of float32.
    for k in attrs:
        v = attrs[k]
        if isinstance(v, numpy.ndarray) and v.dtype == numpy.float32:
            attrs[k] = v.astype(numpy.float64)
    container.add_node(op_type, input_name, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.feature.StandardScalerModel', convert_sparkml_scaler)
register_converter('pyspark.ml.feature.MaxAbsScalerModel', convert_sparkml_scaler)
register_converter('pyspark.ml.feature.MinMaxScalerModel', convert_sparkml_scaler)


def calculate_sparkml_scaler_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])

    input_shape = copy.deepcopy(operator.inputs[0].type.shape)
    operator.outputs[0].type = FloatTensorType(input_shape)


register_shape_calculator('pyspark.ml.feature.StandardScalerModel', calculate_sparkml_scaler_output_shapes)
register_shape_calculator('pyspark.ml.feature.MaxAbsScalerModel', calculate_sparkml_scaler_output_shapes)
register_shape_calculator('pyspark.ml.feature.MinMaxScalerModel', calculate_sparkml_scaler_output_shapes)
