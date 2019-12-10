"""
Tests scilit-learn's tree-based methods' converters.
"""
import sys
import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from onnxmltools.convert import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


def _fit_classification_model(model, n_classes, is_str=False):
    x, y = make_classification(n_classes=n_classes, n_features=100,
                               n_samples=1000,
                               random_state=42, n_informative=7)
    y = y.astype(np.str) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                   random_state=42)
    model.fit(x_train, y_train)
    return model, x_test.astype(np.float32)


class TestScikitLearnModels(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="sklearn converter not tested on python 2")
    def test_sklearn_regressor(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        xgb = RandomForestRegressor()
        xgb.fit(x_train, y_train)
        conv_model = convert_sklearn(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, x_test.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnRFRegressor-Dec3",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="sklearn converter not tested on python 2")
    def test_sklearn_classifier(self):
        xgb, x_test = _fit_classification_model(RandomForestClassifier(), 2)
        conv_model = convert_sklearn(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, x_test.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnRFClassifier",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="sklearn converter not tested on python 2")
    def test_sklearn_classifier_multi(self):
        xgb, x_test = _fit_classification_model(RandomForestClassifier(), 3)
        conv_model = convert_sklearn(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, x_test.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnRFClassifierMulti",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="sklearn converter not tested on python 2")
    def test_sklearn_classifier_multi_str_labels(self):
        xgb, x_test = _fit_classification_model(
            RandomForestClassifier(n_estimators=4), 5, is_str=True)
        conv_model = convert_sklearn(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, x_test.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnRFClassifierMultiStrLabels",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="sklearn converter not tested on python 2")
    def test_sklearn_classifier_multi_discrete_int_labels(self):
        iris = load_iris()
        x = iris.data[:, :2]
        y = iris.target
        y[y == 0] = 10
        y[y == 1] = 20
        y[y == 2] = -30
        x_train, x_test, y_train, _ = train_test_split(x,
                                                       y,
                                                       test_size=0.5,
                                                       random_state=42)
        xgb = RandomForestClassifier(n_estimators=3)
        xgb.fit(x_train, y_train)
        conv_model = convert_sklearn(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, x_test.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnRFClassifierMultiDiscreteIntLabels",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )


if __name__ == "__main__":
    unittest.main()
