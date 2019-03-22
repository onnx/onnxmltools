# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxutils.onnxtk.common._registration import register_shape_calculator
from onnxutils.onnxtk.common.shape_calculator import calculate_linear_regressor_output_shapes

register_shape_calculator('LgbmRegressor', calculate_linear_regressor_output_shapes)
