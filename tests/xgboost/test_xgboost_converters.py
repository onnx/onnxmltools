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
    load_diabetes,
    load_iris,
    make_classification,
    load_digits,
    make_regression,
)
from sklearn.model_selection import train_test_split
from xgboost import (
    XGBRegressor,
    XGBClassifier,
    train,
    DMatrix,
    Booster,
    train as train_xgb,
)
from sklearn.preprocessing import StandardScaler
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from onnxruntime import InferenceSession


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def fct_cl2(y):
    y[y == 2] = 0
    return y


def fct_cl3(y):
    y[y == 0] = 6
    return y


def fct_id(y):
    return y


def _fit_classification_model(model, n_classes, is_str=False, dtype=None):
    x, y = make_classification(
        n_classes=n_classes,
        n_features=100,
        n_samples=1000,
        random_state=42,
        n_informative=7,
    )
    y = y.astype(np.str_) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5, random_state=42)
    if dtype is not None:
        y_train = y_train.astype(dtype)
    model.fit(x_train, y_train)
    return model, x_test.astype(np.float32)


class TestXGBoostModels(unittest.TestCase):
    def test_xgb_regressor(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )
        xgb = XGBRegressor()
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBRegressor-Dec3",
        )

    def test_xgb_regressor_poisson(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target / 100
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=17
        )
        for nest in [5, 50]:
            xgb = XGBRegressor(
                objective="count:poisson",
                random_state=5,
                max_depth=3,
                n_estimators=nest,
            )
            xgb.fit(x_train, y_train)
            conv_model = convert_xgboost(
                xgb,
                initial_types=[("input", FloatTensorType(shape=[None, None]))],
                target_opset=TARGET_OPSET,
            )
            dump_data_and_model(
                x_test.astype("float32"),
                xgb,
                conv_model,
                basename=f"SklearnXGBRegressorPoisson{nest}-Dec3",
            )

    def test_xgb0_classifier(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 2)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(x_test, xgb, conv_model, basename="SklearnXGBClassifier")

    def test_xgb_classifier_uint8(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 2, dtype=np.uint8)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=["None", "None"]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(x_test, xgb, conv_model, basename="SklearnXGBClassifier")

    def test_xgb_classifier_multi(self):
        xgb, x_test = _fit_classification_model(XGBClassifier(), 3)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            x_test, xgb, conv_model, basename="SklearnXGBClassifierMulti"
        )

    def test_xgb_classifier_multi_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective="reg:logistic"), 4
        )
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            x_test, xgb, conv_model, basename="SklearnXGBClassifierMultiRegLog"
        )

    def test_xgb_classifier_reglog(self):
        xgb, x_test = _fit_classification_model(
            XGBClassifier(objective="reg:logistic"), 2
        )
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            x_test, xgb, conv_model, basename="SklearnXGBClassifierRegLog-Dec4"
        )

    def test_xgb_classifier_multi_discrete_int_labels(self):
        iris = load_iris()
        x = iris.data[:, :2]
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )
        xgb = XGBClassifier(n_estimators=3)
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBClassifierMultiDiscreteIntLabels",
        )

    def test_xgb1_booster_classifier_bin(self):
        x, y = make_classification(
            n_classes=2, n_features=5, n_samples=100, random_state=42, n_informative=3
        )
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )

        data = DMatrix(x_train, label=y_train)
        model = train(
            {"objective": "binary:logistic", "n_estimators": 3, "min_child_samples": 1},
            data,
        )
        model_onnx = convert_xgboost(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, x.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test.astype(np.float32), model, model_onnx, basename="XGBBoosterMCl"
        )

    def test_xgb0_booster_classifier_multiclass_softprob(self):
        x, y = make_classification(
            n_classes=3, n_features=5, n_samples=100, random_state=42, n_informative=3
        )
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )

        data = DMatrix(x_train, label=y_train)
        model = train(
            {
                "objective": "multi:softprob",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_class": 3,
            },
            data,
        )
        model_onnx = convert_xgboost(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, x.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test.astype(np.float32),
            model,
            model_onnx,
            basename="XGBBoosterMClSoftProb",
        )

    def test_xgboost_booster_classifier_multiclass_softmax(self):
        x, y = make_classification(
            n_classes=3, n_features=5, n_samples=100, random_state=42, n_informative=3
        )
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )

        data = DMatrix(x_train, label=y_train)
        model = train(
            {
                "objective": "multi:softmax",
                "n_estimators": 3,
                "min_child_samples": 1,
                "num_class": 3,
            },
            data,
        )
        model_onnx = convert_xgboost(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, x.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test.astype(np.float32),
            model,
            model_onnx,
            basename="XGBBoosterMClSoftMax",
        )

    def test_xgboost_booster_reg(self):
        x, y = make_classification(
            n_classes=2, n_features=5, n_samples=100, random_state=42, n_informative=3
        )
        y = y.astype(np.float32) + 0.567
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )

        data = DMatrix(x_train, label=y_train)
        model = train(
            {
                "objective": "reg:squarederror",
                "n_estimators": 3,
                "min_child_samples": 1,
            },
            data,
        )
        model_onnx = convert_xgboost(
            model,
            "tree-based classifier",
            [("input", FloatTensorType([None, x.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test.astype(np.float32), model, model_onnx, basename="XGBBoosterReg"
        )

    def test_xgboost_10(self):
        this = os.path.abspath(os.path.dirname(__file__))
        train = os.path.join(this, "input_fail_train.csv")
        test = os.path.join(this, "input_fail_test.csv")

        param_distributions = {
            "colsample_bytree": 0.5,
            "gamma": 0.2,
            "learning_rate": 0.3,
            "max_depth": 2,
            "min_child_weight": 1.0,
            "n_estimators": 1,
            "missing": np.nan,
        }

        train_df = pandas.read_csv(train)
        X_train, y_train = (
            train_df.drop("label", axis=1).values,
            train_df["label"].fillna(0).values,
        )
        test_df = pandas.read_csv(test)
        X_test, _ = (
            test_df.drop("label", axis=1).values,
            test_df["label"].fillna(0).values,
        )

        regressor = XGBRegressor(
            verbose=0, objective="reg:squarederror", **param_distributions
        )
        regressor.fit(X_train, y_train)

        model_onnx = convert_xgboost(
            regressor,
            "bug",
            [("input", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            X_test.astype(np.float32),
            regressor,
            model_onnx,
            basename="XGBBoosterRegBug",
        )

    def test_xgboost_classifier_i5450_softmax(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
        clr = XGBClassifier(objective="multi:softmax", max_depth=1, n_estimators=2)
        clr.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=40)
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_xgboost(
            clr, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        sess.get_inputs()[0].name
        sess.get_outputs()[1].name
        predict_list = [1.0, 20.0, 466.0, 0.0]
        np.array(predict_list).reshape((1, -1)).astype(np.float32)
        bst = clr.get_booster()
        bst.dump_model("dump.raw.txt")
        dump_data_and_model(
            X_test.astype(np.float32) + 1e-5,
            clr,
            onx,
            basename="XGBClassifierIris-Out0",
        )

    def test_xgboost_classifier_i5450(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
        clr = XGBClassifier(objective="multi:softprob", max_depth=1, n_estimators=2)
        clr.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=40)
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_xgboost(
            clr, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        sess.get_inputs()[0].name
        sess.get_outputs()[1].name
        predict_list = [1.0, 20.0, 466.0, 0.0]
        np.array(predict_list).reshape((1, -1)).astype(np.float32)
        bst = clr.get_booster()
        bst.dump_model("dump.raw.txt")
        dump_data_and_model(
            X_test.astype(np.float32) + 1e-5, clr, onx, basename="XGBClassifierIris"
        )

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
            clf,
            initial_types=[("input", FloatTensorType(sh))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(
            X_test.astype(np.float32), clf, onnx_model, basename="XGBoostExample"
        )

    def test_xgb0_empty_tree_classifier(self):
        xgb = XGBClassifier(n_estimators=2, max_depth=2)

        # simple dataset
        X = [[0, 1], [1, 1], [2, 0]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0]
        xgb.fit(X, y)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            conv_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])

    def test_xgb_best_tree_limit_classifier(self):
        # Train
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        dtrain = DMatrix(X_train, label=y_train)
        dtest = DMatrix(X_test)
        param = {"objective": "multi:softmax", "num_class": 3}
        bst_original = train_xgb(param, dtrain, 10)
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        bst_original.save_model("model.json")

        onx_loaded = convert_xgboost(
            bst_original, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        sess = InferenceSession(
            onx_loaded.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"float_input": X_test.astype(np.float32)})
        assert_almost_equal(
            bst_original.predict(dtest, output_margin=True), res[1], decimal=5
        )
        assert_almost_equal(bst_original.predict(dtest), res[0])

        # After being restored, the loaded booster is not exactly the same
        # in memory. `best_ntree_limit` is not saved during `save_model`.
        bst_loaded = Booster()
        bst_loaded.load_model("model.json")
        bst_loaded.save_model("model2.json")
        assert_almost_equal(
            bst_loaded.predict(dtest, output_margin=True),
            bst_original.predict(dtest, output_margin=True),
            decimal=5,
        )
        assert_almost_equal(bst_loaded.predict(dtest), bst_original.predict(dtest))

        onx_loaded = convert_xgboost(
            bst_loaded, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        sess = InferenceSession(
            onx_loaded.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"float_input": X_test.astype(np.float32)})
        assert_almost_equal(
            bst_loaded.predict(dtest, output_margin=True), res[1], decimal=5
        )
        assert_almost_equal(bst_loaded.predict(dtest), res[0])

    def test_xgb_classifier(self):
        x = np.random.randn(100, 10).astype(np.float32)
        y = ((x.sum(axis=1) + np.random.randn(x.shape[0]) / 50 + 0.5) >= 0).astype(
            np.int64
        )
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        bmy = np.mean(y_train)

        for bm, n_est in [(None, 1), (None, 3), (bmy, 1), (bmy, 3)]:
            model_skl = XGBClassifier(
                n_estimators=n_est,
                learning_rate=0.01,
                subsample=0.5,
                objective="binary:logistic",
                base_score=bm,
                max_depth=2,
            )
            model_skl.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)

            model_onnx_skl = convert_xgboost(
                model_skl,
                initial_types=[("X", FloatTensorType([None, x.shape[1]]))],
                target_opset=TARGET_OPSET,
            )
            with self.subTest(base_score=bm, n_estimators=n_est):
                oinf = InferenceSession(
                    model_onnx_skl.SerializeToString(),
                    providers=["CPUExecutionProvider"],
                )
                res2 = oinf.run(None, {"X": x_test})
                assert_almost_equal(model_skl.predict_proba(x_test), res2[1])

    def test_xgb_cost(self):
        obj_classes = {
            "reg:logistic": (
                XGBClassifier,
                fct_cl2,
                make_classification(n_features=4, n_classes=2, n_clusters_per_class=1),
            ),
            "binary:logistic": (
                XGBClassifier,
                fct_cl2,
                make_classification(n_features=4, n_classes=2, n_clusters_per_class=1),
            ),
            "multi:softmax": (
                XGBClassifier,
                fct_id,
                make_classification(n_features=4, n_classes=3, n_clusters_per_class=1),
            ),
            "multi:softprob": (
                XGBClassifier,
                fct_id,
                make_classification(n_features=4, n_classes=3, n_clusters_per_class=1),
            ),
            "reg:squarederror": (
                XGBRegressor,
                fct_id,
                make_regression(n_features=4, n_targets=1),
            ),
            "reg:squarederror2": (
                XGBRegressor,
                fct_id,
                make_regression(n_features=4, n_targets=2),
            ),
        }
        nb_tests = 0
        for objective in obj_classes:  # pylint: disable=C0206
            for n_estimators in [1, 2]:
                with self.subTest(objective=objective, n_estimators=n_estimators):
                    probs = []
                    cl, fct, prob = obj_classes[objective]

                    iris = load_iris()
                    X, y = iris.data, iris.target
                    y = fct(y)
                    X_train, X_test, y_train, _ = train_test_split(
                        X, y, random_state=11
                    )
                    probs.append((X_train, X_test, y_train))

                    X_train, X_test, y_train, _ = train_test_split(
                        *prob, random_state=11
                    )
                    probs.append((X_train, X_test, y_train))

                    for X_train, X_test, y_train in probs:
                        obj = objective.replace("reg:squarederror2", "reg:squarederror")
                        obj = obj.replace("multi:softmax2", "multi:softmax")
                        clr = cl(objective=obj, n_estimators=n_estimators)
                        if len(y_train.shape) == 2:
                            y_train = y_train[:, 1]
                        try:
                            clr.fit(X_train, y_train)
                        except ValueError as e:
                            raise AssertionError(
                                "Unable to train with objective %r and data %r."
                                % (objective, y_train)
                            ) from e

                        model_def = convert_xgboost(
                            clr,
                            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
                            target_opset=TARGET_OPSET,
                        )

                        oinf = InferenceSession(
                            model_def.SerializeToString(),
                            providers=["CPUExecutionProvider"],
                        )
                        y = oinf.run(None, {"X": X_test.astype(np.float32)})
                        if cl == XGBRegressor:
                            exp = clr.predict(X_test)
                            assert_almost_equal(exp, y[0].ravel(), decimal=5)
                        else:
                            if "softmax" not in obj:
                                exp = clr.predict_proba(X_test)
                                got = pandas.DataFrame(y[1]).values
                                assert_almost_equal(exp, got, decimal=5)

                            exp = clr.predict(X_test[:10])
                            assert_almost_equal(exp, y[0][:10])

                        nb_tests += 1

        self.assertGreater(nb_tests, 8)

    def test_xgb_classifier_601(self):
        model = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            gamma=0,
            importance_type="gain",
            interaction_constraints="",
            learning_rate=0.3,
            max_delta_step=0,
            max_depth=6,
            min_child_weight=1,
            missing=np.nan,
            n_estimators=3,
            n_jobs=0,
            num_parallel_tree=1,
            objective="multi:softprob",
            random_state=0,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=None,
            subsample=1,
            tree_method="exact",
            validate_parameters=1,
        )
        xgb, x_test = _fit_classification_model(model, 3)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        dump_data_and_model(x_test, xgb, conv_model, basename="SklearnXGBClassifier601")

    def test_xgb_classifier_hinge(self):
        model = XGBClassifier(
            n_estimators=3, objective="binary:hinge", random_state=0, max_depth=2
        )
        xgb, x_test = _fit_classification_model(model, 2)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test, xgb, conv_model, basename="SklearnXGBClassifierHinge"
        )

    def test_doc_example(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X = X.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = XGBClassifier()
        clr.fit(X_train, y_train)
        expected_prob = clr.predict_proba(X_test)

        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_xgboost(clr, initial_types=initial_type)

        sess = InferenceSession(onx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: X_test.astype(np.float32)})
        assert_almost_equal(expected_prob, pred_onx[1], decimal=5)

        dtrain = DMatrix(X_train, label=y_train)
        dtest = DMatrix(X_test)
        param = {"objective": "multi:softmax", "num_class": 3}
        bst = train_xgb(param, dtrain, 10)
        expected_prob = bst.predict(dtest, output_margin=True)
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onx = convert_xgboost(bst, initial_types=initial_type)
        sess = InferenceSession(onx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: X_test.astype(np.float32)})
        assert_almost_equal(expected_prob, pred_onx[1], decimal=5)

    def test_xgb_classifier_13(self):
        this = os.path.dirname(__file__)
        df = pandas.read_csv(os.path.join(this, "data_fail_empty.csv"))
        X, y = df.drop("y", axis=1), df["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clr = XGBClassifier(
            max_delta_step=0,
            tree_method="hist",
            n_estimators=100,
            booster="gbtree",
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.1,
            gamma=10,
            max_depth=7,
            min_child_weight=50,
            subsample=0.75,
            colsample_bytree=0.75,
            random_state=42,
            verbosity=0,
        )

        clr.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=40)

        initial_type = [("float_input", FloatTensorType([None, 797]))]
        onx = convert_xgboost(
            clr, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        expected = clr.predict(X_test), clr.predict_proba(X_test)
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        X_test = X_test.values.astype(np.float32)
        got = sess.run(None, {"float_input": X_test})
        assert_almost_equal(expected[1], got[1])
        assert_almost_equal(expected[0], got[0])

    def test_xgb_classifier_13_2(self):
        this = os.path.dirname(__file__)
        df = pandas.read_csv(os.path.join(this, "data_bug.csv"))
        X, y = df.drop("y", axis=1), df["y"]
        x_train, x_test, y_train, y_test = train_test_split(
            X.values.astype(np.float32), y.values.astype(np.float32), random_state=2022
        )

        model_param = {
            "objective": "binary:logistic",
            "n_estimators": 1000,
            "early_stopping_rounds": 113,
            "random_state": 42,
            "max_depth": 3,
        }
        eval_metric = ["logloss", "auc", "error"]
        model = XGBClassifier(**model_param)
        model.fit(
            X=x_train,
            y=y_train,
            eval_set=[(x_test, y_test)],
            eval_metric=eval_metric,
            verbose=False,
        )

        initial_types = [("float_input", FloatTensorType([None, x_train.shape[1]]))]
        onnx_model = convert_xgboost(model, initial_types=initial_types)
        for att in onnx_model.graph.node[0].attribute:
            if att.name == "nodes_treeids":
                self.assertLess(max(att.ints), 1000)
            if att.name == "class_ids":
                self.assertEqual(set(att.ints), {0})
            if att.name == "base_values":
                self.assertEqual(len(att.floats), 1)
            if att.name == "post_transform":
                self.assertEqual(att.s, b"LOGISTIC")

        expected = model.predict(x_test), model.predict_proba(x_test)
        sess = InferenceSession(onnx_model.SerializeToString())
        got = sess.run(None, {"float_input": x_test})
        assert_almost_equal(expected[1], got[1])
        assert_almost_equal(expected[0], got[0])


if __name__ == "__main__":
    TestXGBoostModels().test_xgb_classifier_13_2()
    unittest.main(verbosity=2)
