"""
Tests CatBoost CatBoostRegressor converter.
"""
import unittest
import numpy
import catboost
import onnxruntime as rt
from catboost.utils import convert_to_onnx_object
from sklearn.datasets import make_regression, make_classification
from onnxmltools.convert import convert_catboost
from onnxmltools.utils import dump_data_and_model, dump_single_regression, dump_multiple_classification
from onnxmltools.utils.tests_helper import convert_model


class TestCatBoost(unittest.TestCase):
    def test_catboost_regressor(self):
        X, y = make_regression(n_samples=100, n_features=4, random_state=0)
        catboost_model = catboost.CatBoostRegressor(task_type='CPU', loss_function='RMSE',
                                                    n_estimators=10, verbose=0)
        dump_single_regression(catboost_model)

        catboost_model.fit(X.astype(numpy.float32), y)
        catboost_onnx = convert_catboost(catboost_model, name='CatBoostRegression',
                                         doc_string='test regression')
        self.assertTrue(catboost_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), catboost_model, catboost_onnx, basename="CatBoostReg-Dec4")

    def test_catboost_bin_classifier(self):
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        catboost_model = catboost.CatBoostClassifier(task_type='CPU', loss_function='MultiClass',
                                                     n_estimators=10, verbose=0)

        catboost_model.fit(X.astype(numpy.float32), y)

        catboost_onnx = convert_catboost(catboost_model, name='CatBoostBinClassification',
                                         doc_string='test binary classification')
        self.assertTrue(catboost_onnx is not None)
        # onnx runtime returns zeros as class labels
        # dump_data_and_model(X.astype(numpy.float32), catboost_model, catboost_onnx, basename="CatBoostBinClass")

    def test_catboost_multi_classifier(self):
        X, y = make_classification(n_samples=10, n_informative=8, n_classes=3, random_state=0)
        catboost_model = catboost.CatBoostClassifier(task_type='CPU', loss_function='MultiClass', n_estimators=100,
                                                    verbose=0)

        dump_multiple_classification(catboost_model)

        catboost_model.fit(X.astype(numpy.float32), y)
        catboost_onnx = convert_catboost(catboost_model, name='CatBoostMultiClassification',
                                         doc_string='test multiclass classification')
        self.assertTrue(catboost_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), catboost_model, catboost_onnx, basename="CatBoostMultiClass")


if __name__ == "__main__":
    unittest.main()