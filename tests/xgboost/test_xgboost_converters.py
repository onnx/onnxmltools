"""
Tests scilit-learn's tree-based methods' converters.
"""
import os
import sys
import unittest
import numpy as np
import pandas
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier, train, DMatrix
from onnxmltools.convert import convert_xgboost
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


class TestXGBoostModels(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_regressor(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        xgb = XGBRegressor()
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBRegressor-Dec3",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 2)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnXGBClassifier",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 3)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMulti",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective='reg:logistic'), 4)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiRegLog",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective='reg:logistic'), 2)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnXGBClassifierRegLog",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_str_labels(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(n_estimators=4), 5, is_str=True)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test,
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiStrLabels",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_classifier_multi_discrete_int_labels(self):
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
        xgb = XGBClassifier(n_estimators=3)
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=['None', 'None']))])
        self.assertTrue(conv_model is not None)
        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiDiscreteIntLabels",
            allow_failure="StrictVersion("
            "onnx.__version__)"
            "< StrictVersion('1.3.0')",
        )

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgboost_booster_classifier_bin(self):
        x, y = make_classification(n_classes=2, n_features=5,
                                   n_samples=100,
                                   random_state=42, n_informative=3)
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        
        data = DMatrix(x_train, label=y_train)
        model = train({'objective': 'binary:logistic',
                       'n_estimators': 3, 'min_child_samples': 1}, data)
        model_onnx = convert_xgboost(model, 'tree-based classifier',
                                     [('input', FloatTensorType([None, x.shape[1]]))])
        dump_data_and_model(x_test.astype(np.float32),
                            model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename="XGBBoosterMCl")

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgboost_booster_classifier_multiclass(self):
        x, y = make_classification(n_classes=3, n_features=5,
                                   n_samples=100,
                                   random_state=42, n_informative=3)
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        
        data = DMatrix(x_train, label=y_train)
        model = train({'objective': 'multi:softprob',
                       'n_estimators': 3, 'min_child_samples': 1,
                       'num_class': 3}, data)
        model_onnx = convert_xgboost(model, 'tree-based classifier',
                                     [('input', FloatTensorType([None, x.shape[1]]))])
        dump_data_and_model(x_test.astype(np.float32),
                            model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename="XGBBoosterMCl")

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgboost_booster_classifier_reg(self):
        x, y = make_classification(n_classes=2, n_features=5,
                                   n_samples=100,
                                   random_state=42, n_informative=3)        
        y = y.astype(np.float32) + 0.567
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        
        data = DMatrix(x_train, label=y_train)
        model = train({'objective': 'reg:squarederror',
                       'n_estimators': 3, 'min_child_samples': 1}, data)
        model_onnx = convert_xgboost(model, 'tree-based classifier',
                                     [('input', FloatTensorType([None, x.shape[1]]))])
        dump_data_and_model(x_test.astype(np.float32),
                            model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename="XGBBoosterReg")

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgboost_10(self):
        this = os.path.abspath(os.path.dirname(__file__))
        train = os.path.join(this, "input_fail_train.csv")
        test = os.path.join(this, "input_fail_test.csv")
        
        param_distributions = {
            "colsample_bytree": 0.5,
            "gamma": 0.2,
            'learning_rate': 0.3,
            'max_depth': 2,
            'min_child_weight': 1.,
            'n_estimators': 1,
            'missing': np.nan,
        }
        
        train_df = pandas.read_csv(train)
        X_train, y_train = train_df.drop('label', axis=1).values, train_df['label'].values
        test_df = pandas.read_csv(test)
        X_test, y_test = test_df.drop('label', axis=1).values, test_df['label'].values
        
        regressor = XGBRegressor(verbose=0, objective='reg:squarederror', **param_distributions)
        regressor.fit(X_train, y_train)
        
        model_onnx = convert_xgboost(
            regressor, 'bug',
            [('input', FloatTensorType([None, X_train.shape[1]]))])

        dump_data_and_model(
            X_test.astype(np.float32),
            regressor, model_onnx,
            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
            basename="XGBBoosterRegBug")


if __name__ == "__main__":
    unittest.main()
