# SPDX-License-Identifier: Apache-2.0

"""
Tests scilit-learn's tree-based methods' converters.
"""
import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas
from sklearn.datasets import (
    load_diabetes, load_iris, make_classification, load_digits)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier, train, DMatrix
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from onnxruntime import InferenceSession


def _fit_classification_model(model, n_classes, is_str=False, dtype=None):
    x, y = make_classification(n_classes=n_classes, n_features=100,
                               n_samples=1000,
                               random_state=42, n_informative=7)
    y = y.astype(np.str) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                   random_state=42)
    if dtype is not None:
        y_train = y_train.astype(dtype)
    model.fit(x_train, y_train)
    return model, x_test.astype(np.float32)


class TestXGBoostModels(unittest.TestCase):

    def test_xgb_regressor(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        xgb = XGBRegressor()
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgb_classifier(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 2)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgb_classifier_uint8(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(), 2, dtype=np.uint8)
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

    def test_xgb_classifier_multi(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 3)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgb_classifier_multi_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective='reg:logistic'), 4)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgb_classifier_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective='reg:logistic'), 2)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgb_classifier_multi_str_labels(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(n_estimators=4), 5, is_str=True)
        conv_model = convert_xgboost(
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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
            xgb, initial_types=[('input', FloatTensorType(shape=[None, None]))])
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

    def test_xgboost_booster_classifier_multiclass_softprob(self):
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
                            basename="XGBBoosterMClSoftProb")

    def test_xgboost_booster_classifier_multiclass_softmax(self):
        x, y = make_classification(n_classes=3, n_features=5,
                                   n_samples=100,
                                   random_state=42, n_informative=3)
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5,
                                                       random_state=42)
        
        data = DMatrix(x_train, label=y_train)
        model = train({'objective': 'multi:softmax',
                       'n_estimators': 3, 'min_child_samples': 1,
                       'num_class': 3}, data)
        model_onnx = convert_xgboost(model, 'tree-based classifier',
                                     [('input', FloatTensorType([None, x.shape[1]]))])
        dump_data_and_model(x_test.astype(np.float32),
                            model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename="XGBBoosterMClSoftMax")

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

    def test_xgboost_classifier_i5450(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
        clr = XGBClassifier(objective="multi:softmax", max_depth=1, n_estimators=2)
        clr.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=40)
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_xgboost(clr, initial_types=initial_type)
        sess = InferenceSession(onx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name
        predict_list = [1.,  20., 466.,   0.]
        predict_array = np.array(predict_list).reshape((1,-1)).astype(np.float32)
        pred_onx = sess.run([label_name], {input_name: predict_array})[0]
        pred_xgboost = sessresults=clr.predict_proba(predict_array)
        bst = clr.get_booster()
        bst.dump_model('dump.raw.txt')
        dump_data_and_model(
            X_test.astype(np.float32) + 1e-5,
            clr, onx,
            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
            basename="XGBClassifierIris")

    def test_xgboost_example_mnist(self):
        """
        Train a simple xgboost model and store associated artefacts.
        """
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = XGBClassifier(objective="multi:softprob", n_jobs=-1)
        clf.fit(X_train, y_train)

        sh = [None, X_train.shape[1]]
        onnx_model = convert_xgboost(
            clf, initial_types=[('input', FloatTensorType(sh))])

        dump_data_and_model(
            X_test.astype(np.float32), clf, onnx_model,
            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
            basename="XGBoostExample")

    def test_xgb_empty_tree(self):
        xgb = XGBClassifier(n_estimators=2, max_depth=2)

        # simple dataset
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0]
        xgb.fit(X, y)
        conv_model = convert_xgboost(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        sess = InferenceSession(conv_model.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])


if __name__ == "__main__":
    unittest.main()
