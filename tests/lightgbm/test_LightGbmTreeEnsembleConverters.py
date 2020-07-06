# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion

import lightgbm
import numpy
from lightgbm import LGBMClassifier, LGBMRegressor
import onnxruntime
from onnxmltools.convert.common.utils import hummingbird_installed
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from onnxmltools.utils import dump_binary_classification, dump_multiple_classification
from onnxmltools.utils import dump_single_regression
from onnxmltools.utils.tests_helper import convert_model


class TestLightGbmTreeEnsembleModels(unittest.TestCase):

    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        dump_binary_classification(model, allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")
        dump_multiple_classification(model, allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")

    def test_lightgbm_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        onx = convert_model(
            model, 'dummy', input_types=[('X', FloatTensorType([None, X.shape[1]]))])
        assert "zipmap" in str(onx).lower()

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        dump_single_regression(model)

    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        dump_single_regression(model, suffix="1")

    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        dump_single_regression(model, suffix="2")

    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'binary',
                                'n_estimators': 3, 'min_child_samples': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)

    def test_lightgbm_booster_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'binary',
                                'n_estimators': 3, 'min_child_samples': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))])
        assert "zipmap" in str(model_onnx).lower()
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)

    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'multiclass',
                                'n_estimators': 3, 'min_child_samples': 1, 'num_class': 3},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)
        try:
            from onnxruntime import InferenceSession
        except ImportError:
            # onnxruntime not installed (python 2.7)
            return
        sess = InferenceSession(model_onnx.SerializeToString())
        out = sess.get_outputs()
        names = [o.name for o in out]
        assert names == ['label', 'probabilities']

    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 1.1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'regression',
                                'n_estimators': 3, 'min_child_samples': 1, 'max_depth': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based binary classifier',
                                           [('input', FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "BoosterBin" + model.__class__.__name__)
    
    # Tests with ONNX operators only
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'binary',
                                'n_estimators': 3, 'min_child_samples': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))], without_onnx_ml=True)
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'binary',
                                'n_estimators': 3, 'min_child_samples': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))], without_onnx_ml=True)
        assert "zipmap" in str(model_onnx).lower()
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'multiclass',
                                'n_estimators': 3, 'min_child_samples': 1, 'num_class': 3},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based classifier',
                                           [('input', FloatTensorType([None, 2]))], without_onnx_ml=True)
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)
        try:
            from onnxruntime import InferenceSession
        except ImportError:
            # onnxruntime not installed (python 2.7)
            return
        sess = InferenceSession(model_onnx.SerializeToString())
        out = sess.get_outputs()
        names = [o.name for o in out]
        assert names == ['label', 'probabilities']

    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 1.1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'regression',
                                'n_estimators': 3, 'min_child_samples': 1, 'max_depth': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based binary classifier',
                                           [('input', FloatTensorType([None, 2]))], without_onnx_ml=True)
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.0.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)
    
    # Base test implementation comparing ONNXML and ONNX models.
    def _test_lgbm(self, X, model, extra_config={}):
        # Create ONNX-ML model
        onnx_ml_model = convert_model(
            model, 'lgbm-onnxml', [("input", FloatTensorType([X.shape[0], X.shape[1]]))]
        )[0]

        # Create ONNX model
        onnx_model = convert_model(
            model, 'lgbm-onnx', [("input", FloatTensorType([X.shape[0], X.shape[1]]))], without_onnx_ml=True
        )[0]

        try:
            from onnxruntime import InferenceSession
        except ImportError:
            # onnxruntime not installed (python 2.7)
            return
            
        # Get the predictions for the ONNX-ML model
        session = InferenceSession(onnx_ml_model.SerializeToString())
        output_names = [session.get_outputs()[i].name for i in range(len(session.get_outputs()))]
        onnx_ml_pred = [[] for i in range(len(output_names))]
        inputs = {session.get_inputs()[0].name: X}
        pred = session.run(output_names, inputs)
        for i in range(len(output_names)):
            if output_names[i] == "label":
                onnx_ml_pred[1] = pred[i]
            else:
                onnx_ml_pred[0] = pred[i]

        # Get the predictions for the ONNX model
        session = InferenceSession(onnx_model.SerializeToString())
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
        numpy.testing.assert_allclose(onnx_ml_pred[0], onnx_pred[0], rtol=rtol, atol=atol)

    # Utility function for testing classification models.
    def _test_classifier(self, X, model, rtol=1e-06, atol=1e-06, extra_config={}):
        onnx_ml_pred, onnx_pred, output_names = self._test_lgbm(X, model, extra_config)

        numpy.testing.assert_allclose(onnx_ml_pred[1], onnx_pred[1], rtol=rtol, atol=atol)  # labels
        numpy.testing.assert_allclose(
            list(map(lambda x: list(x.values()), onnx_ml_pred[0])), onnx_pred[0], rtol=rtol, atol=atol
        )  # probs

    # Regression test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 1 estimator.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with 2 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        self._test_regressor(X, model)

    # Regression test with gbdt boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 1.1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {"boosting_type": "gbdt", "objective": "regression", "n_estimators": 3, "min_child_samples": 1, "max_depth": 1},
            data,
        )
        self._test_regressor(X, model)

    # Binary classification test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators zipmap.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Binary classification test with 3 estimators and selecting boosting type zipmap.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_classifier_zipmap(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({"boosting_type": "gbdt", "objective": "binary", "n_estimators": 3, "min_child_samples": 1}, data)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_classifier_multi(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        self._test_classifier(X, model)

    # Multiclass classification test with 3 estimators and selecting boosting type.
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train(
            {"boosting_type": "gbdt", "objective": "multiclass", "n_estimators": 3, "min_child_samples": 1, "num_class": 3},
            data,
        )
        self._test_classifier(X, model)


if __name__ == "__main__":
    unittest.main()
