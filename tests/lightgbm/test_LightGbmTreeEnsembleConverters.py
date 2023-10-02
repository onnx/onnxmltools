# SPDX-License-Identifier: Apache-2.0

import unittest

import lightgbm
import numpy
from numpy.testing import assert_almost_equal
from onnx.defs import onnx_opset_version
from lightgbm import LGBMClassifier, LGBMRegressor
import onnxruntime
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_lightgbm
from onnxmltools.utils import dump_data_and_model
from onnxmltools.utils import dump_binary_classification, dump_multiple_classification
from onnxmltools.utils import dump_single_regression
from onnxmltools.utils.tests_helper import convert_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestLightGbmTreeEnsembleModels(unittest.TestCase):
    def test_lightgbm_classifier_binary(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        dump_binary_classification(model)

    def test_lightgbm_classifier_multiple(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        dump_multiple_classification(model)

    def test_lightgbm_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        model.fit(X, y)
        onx = convert_model(
            model,
            "dummy",
            input_types=[("X", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" in str(onx).lower()

    def test_lightgbm_classifier_nozipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [1, 5], [6, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 1, 0]
        model = LGBMClassifier(
            n_estimators=3, min_child_samples=1, max_depth=2, num_thread=1
        )
        model.fit(X, y)
        onx = convert_model(
            model,
            "dummy",
            input_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" not in str(onx).lower()
        onxs = onx[0].SerializeToString()
        try:
            sess = onnxruntime.InferenceSession(
                onxs, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise AssertionError(
                "Model cannot be loaded by onnxruntime due to %r\n%s." % (e, onx[0])
            )
        exp = model.predict(X), model.predict_proba(X)
        got = sess.run(None, {"X": X})
        assert_almost_equal(exp[0], got[0])
        assert_almost_equal(exp[1], got[1])

    def test_lightgbm_classifier_nozipmap2(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [1, 5], [6, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 1, 0]
        model = LGBMClassifier(
            n_estimators=3, min_child_samples=1, max_depth=2, num_thread=1
        )
        model.fit(X, y)
        onx = convert_lightgbm(
            model,
            "dummy",
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" not in str(onx).lower()
        onxs = onx.SerializeToString()
        try:
            sess = onnxruntime.InferenceSession(
                onxs, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise AssertionError(
                "Model cannot be loaded by onnxruntime due to %r\n%s." % (e, onx)
            )
        exp = model.predict(X), model.predict_proba(X)
        got = sess.run(None, {"X": X})
        assert_almost_equal(exp[0], got[0])
        assert_almost_equal(exp[1], got[1])

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1, num_thread=1)
        dump_single_regression(model)

    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1, num_thread=1)
        dump_single_regression(model, suffix="1")

    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(
            n_estimators=2, max_depth=1, min_child_samples=1, num_thread=1
        )
        dump_single_regression(model, suffix="2")

    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_thread": 1,
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    def test_lightgbm_booster_classifier_nozipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_thread": 1,
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, 2]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" not in str(model_onnx).lower()
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    def test_lightgbm_booster_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_thread": 1,
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET,
        )
        assert "zipmap" in str(model_onnx).lower()
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_class": 3,
                "num_thread": 1,
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )
        try:
            from onnxruntime import InferenceSession
        except ImportError:
            # onnxruntime not installed (python 2.7)
            return
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        out = sess.get_outputs()
        names = [o.name for o in out]
        assert names == ["label", "probabilities"]

    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 1.1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "regression",
                "n_estimators": 3,
                "min_child_samples": 1,
                "max_depth": 1,
                "num_thread": 1,
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based binary classifier",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )


if __name__ == "__main__":
    unittest.main()
