"""
Tests scilit-learn's tree-based methods' converters.
"""
import sys
import unittest
from sklearn.datasets import load_iris
from xgboost import XGBRegressor, XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_multiple_classification, dump_single_regression, dump_binary_classification


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


if __name__ == "__main__":
    unittest.main()
