"""
Tests scilit-learn's tree-based methods' converters.
"""
import sys
import unittest
from sklearn.datasets import load_iris
from xgboost import XGBRegressor, XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import (
    dump_data_and_model, dump_binary_classification
    dump_multiple_classification, dump_single_regression,
)


class TestXGBoostModels(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converter not tested on python 2")
    def test_xgb_regressor(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBRegressor()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        dump_single_regression(xgb, suffix="-Dec4")

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converter not tested on python 2")
    def test_xgb_classifier(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        dump_binary_classification(xgb)

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(xgb, allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier(objective='reg:logistic')
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 2]))])
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(xgb, suffix="RegLog",
                                     allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0

        xgb = XGBClassifier(objective='reg:logistic')
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 2]))])
        self.assertTrue(conv_model is not None)
        dump_binary_classification(xgb, suffix="RegLog")

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_str_labels(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target.astype('str')
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.5,
                                                            random_state=42)

        xgb = XGBClassifier(n_estimators=4)
        xgb.fit(X_train, y_train)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            X_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiStrLabels",
            allow_failure="StrictVersion("
            "onnxr.__version__)"
            "<= StrictVersion('0.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_discrete_int_labels(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y==0] = 10
        y[y==1] = 20
        y[y==2] = -30
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.5,
                                                            random_state=42)

        xgb = XGBClassifier(n_estimators=3)
        xgb.fit(X_train, y_train)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            X_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiDiscreteIntLabels",
            allow_failure="StrictVersion("
            "onnxr.__version__)"
            "<= StrictVersion('0.3.0')",
        )


if __name__ == "__main__":
    unittest.main()
