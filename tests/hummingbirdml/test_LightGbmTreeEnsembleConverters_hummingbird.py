# SPDX-License-Identifier: Apache-2.0

import unittest

import lightgbm
import numpy
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxruntime import InferenceSession
from onnxmltools.convert.common.utils import hummingbird_installed
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils.tests_helper import convert_model
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
# PyTorch 1.8.1 supports up to opset version 13.
HUMMINGBIRD_TARGET_OPSET = min(TARGET_OPSET, 13)


class TestLightGbmTreeEnsembleModelsHummingBird(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("BEGIN.")
        import torch

        print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def tearDownClass(cls):
        print("END.")

    # Tests with ONNX operators only
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
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
            without_onnx_ml=True,
            target_opset=HUMMINGBIRD_TARGET_OPSET,
            zipmap=False,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
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
            without_onnx_ml=False,
            target_opset=HUMMINGBIRD_TARGET_OPSET,
        )
        assert "zipmap" in str(model_onnx).lower()
        with self.assertRaises(NotImplementedError):
            convert_model(
                model,
                "tree-based classifier",
                [("input", FloatTensorType([None, 2]))],
                without_onnx_ml=True,
                target_opset=HUMMINGBIRD_TARGET_OPSET,
            )

        model_onnx, prefix = convert_model(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, 2]))],
            without_onnx_ml=True,
            target_opset=HUMMINGBIRD_TARGET_OPSET,
            zipmap=False,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
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
            without_onnx_ml=True,
            target_opset=HUMMINGBIRD_TARGET_OPSET,
            zipmap=False,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        out = sess.get_outputs()
        names = [o.name for o in out]
        assert names == ["label", "probabilities"]

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
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
            },
            data,
        )
        model_onnx, prefix = convert_model(
            model,
            "tree-based binary regressor",
            [("input", FloatTensorType([None, 2]))],
            without_onnx_ml=True,
            target_opset=HUMMINGBIRD_TARGET_OPSET,
            zipmap=False,
        )
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename=prefix + "BoosterBin" + model.__class__.__name__,
        )

    # Base test implementation comparing ONNXML and ONNX models.
    def _test_lgbm(self, X, model, extra_config={}):
        # Create ONNX-ML model
        onnx_ml_model = convert_model(
            model,
            "lgbm-onnxml",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )[0]

        # Create ONNX model
        onnx_model = convert_model(
            model,
            "lgbm-onnx",
            [("input", FloatTensorType([None, X.shape[1]]))],
            without_onnx_ml=True,
            target_opset=TARGET_OPSET,
        )[0]

        # Get the predictions for the ONNX-ML model
        session = InferenceSession(
            onnx_ml_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output_names = [
            session.get_outputs()[i].name for i in range(len(session.get_outputs()))
        ]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        pred = session.run(output_names, inputs)
        for i in range(len(output_names)):
            if output_names[i] == "label":
                onnx_ml_pred[1] = pred[i]
            else:
                onnx_ml_pred[0] = pred[i]

        # Get the predictions for the ONNX model
        session = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_pred = [[] for i in range(len(output_names))]
        pred = session.run(output_names, inputs)
        for i in range(len(output_names)):
            if output_names[i] == "label":
                onnx_pred[1] = pred[i]
            else:
                onnx_pred[0] = pred[i]

        return onnx_ml_pred, onnx_pred, output_names

    # Utility function for testing regression models.
    def _test_regressor(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        # Check that predicted values match
        numpy.testing.assert_allclose(
            onnx_ml_pred[0], onnx_pred[0], rtol=rtol, atol=atol
        )

    # Utility function for testing classification models.
    def _test_classifier(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        numpy.testing.assert_allclose(
            onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol
        )  # labels
        numpy.testing.assert_allclose(
            list(map(lambda x: list(x.values()), onnx_ml_pred[0])),
            onnx_pred[0],
            rtol=rtol,
            atol=atol,
        )  # probs

    # Regression test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_regressor(self):
        X = [[0, 1], [1, 1], [2, 0], [4, 0], [2, 3]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50, 10, 10], dtype=numpy.float32)
        model = LGBMRegressor(n_estimators=3, min_child_samples=1, num_thread=1)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 1 estimator.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1, num_thread=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 2 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_regressor2(self):
        model = LGBMRegressor(
            n_estimators=2, max_depth=1, min_child_samples=1, num_thread=1
        )
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with gbdt boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_booster_regressor(self):
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
        self._test_regressor(X, model)

    # Binary classification test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators zipmap.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_booster_classifier(self):
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
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type zipmap.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_booster_classifier_zipmap(self):
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
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_classifier_multi(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1, num_thread=1)
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators and selecting boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    def _test_lightgbm_booster_multi_classifier(self):
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
        self._test_classifier(X, model)


if __name__ == "__main__":
    unittest.main()
