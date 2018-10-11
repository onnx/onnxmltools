# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion as _StrictVersion
import warnings
import onnxmltools
import numpy as np
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.decomposition import TruncatedSVD

np.random.seed(0)

from .test_single_operator_with_cntk_backend import _create_tensor, _evaluate


class TestSklearn2ONNX(unittest.TestCase):

    def test_truncated_svd(self):
        import keras as _keras
        if _keras.__version__ >= _StrictVersion('2.2.0'):
            try:
                from keras_applications.mobilenet import relu6
            except ImportError as e:
                warnings.warn("Unable to run this test due to {}".format(e))
                return
        N, C, K = 2, 3, 2
        x = _create_tensor(N, C)

        svd = TruncatedSVD(n_components=K)
        svd.fit(x)
        onnx_model = onnxmltools.convert_sklearn(svd, initial_types=[('input', FloatTensorType(shape=[1, C]))])
        y_reference = svd.transform(x)
        y_produced = _evaluate(onnx_model, x)

        self.assertTrue(np.allclose(y_reference, y_produced))
