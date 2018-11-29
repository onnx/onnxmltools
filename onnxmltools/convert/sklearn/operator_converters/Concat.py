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
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler


def convert_sklearn_concat(scope, operator, container):
    op = operator.raw_operator
    transform_result_name = []
    supported_transformer_list = [PCA, TruncatedSVD, Binarizer, Imputer]
    if type(transform) not in supported_transformer_list:
        raise NotImplementedError

    container.add_node('Concat', [s for s in operator.inputs],
                       operator.outputs[-1].full_name, name=scope.get_unique_operator_name('Concat'), axis=1)


register_converter('SklearnConcat', convert_sklearn_concat)
