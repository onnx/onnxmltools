# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers
import six
from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_classifier_output_shapes





register_shape_calculator('SklearnLinearClassifier', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnLinearSVC', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnDecisionTreeClassifier', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnRandomForestClassifier', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnExtraTreesClassifier', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnGradientBoostingClassifier', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnBernoulliNB', calculate_linear_classifier_output_shapes)
register_shape_calculator('SklearnMultinomialNB', calculate_linear_classifier_output_shapes)
