"""
Tests h2o's tree-based methods' converters.
"""
import unittest
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h2o
from h2o import H2OFrame
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

from onnxmltools.convert import convert_h2o
from onnxmltools.utils import dump_data_and_model


def _make_mojo(model, train, y=-1, force_y_numeric=False):
    if y < 0:
        y = train.ncol + y
    if force_y_numeric:
        train[y] = train[y].asnumeric()
    x = list(range(0, train.ncol))
    x.remove(y)
    model.train(x=x, y=y, training_frame=train)
    folder = os.environ.get('ONNXTESTDUMP', 'tests/temp')
    if not os.path.exists(folder):
        os.makedirs(folder)
    return model.download_mojo(path=folder)


def _convert_mojo(mojo_path):
    f = open(mojo_path, "rb")
    mojo_content = f.read()
    f.close()
    return convert_h2o(mojo_content)


def _prepare_one_hot(file, y, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    frame = h2o.import_file(dir_path + "/" + file)
    train, test = frame.split_frame([0.95], seed=42)

    cols_to_encode = []
    other_cols = []
    for name, ctype in test.types.items():
        if name == y or name in exclude_cols:
            pass
        elif ctype == "enum":
            cols_to_encode.append(name)
        else:
            other_cols.append(name)
    train_frame = train.as_data_frame()
    train_encode = train_frame.loc[:, cols_to_encode]
    train_other = train_frame.loc[:, other_cols + [y]]
    enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
    enc.fit(train_encode)
    colnames = []
    for cidx in range(len(cols_to_encode)):
        for val in enc.categories_[cidx]:
            colnames.append(cols_to_encode[cidx] + "." + val)

    train_encoded = enc.transform(train_encode.values).toarray()
    train_encoded = pd.DataFrame(train_encoded)
    train_encoded.columns = colnames
    train = train_other.join(train_encoded)
    train = H2OFrame(train)

    test_frame = test.as_data_frame()
    test_encode = test_frame.loc[:, cols_to_encode]
    test_other = test_frame.loc[:, other_cols]

    test_encoded = enc.transform(test_encode.values).toarray()
    test_encoded = pd.DataFrame(test_encoded)
    test_encoded.columns = colnames
    test = test_other.join(test_encoded)

    return train, test


def _train_test_split_as_frames(x, y, is_str=False, is_classifier=False):
    y = y.astype(np.str) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
    f_train_x = H2OFrame(x_train)
    f_train_y = H2OFrame(y_train)
    f_train = f_train_x.cbind(f_train_y)
    if is_classifier:
        f_train[f_train.ncol - 1] = f_train[f_train.ncol - 1].asfactor()
    return f_train, x_test.astype(np.float32)


def _train_classifier(model, n_classes, is_str=False, force_y_numeric=False):
    x, y = make_classification(
        n_classes=n_classes, n_features=100, n_samples=1000,
        random_state=42, n_informative=7
    )
    train, test = _train_test_split_as_frames(x, y, is_str, is_classifier=True)
    mojo_path = _make_mojo(model, train, force_y_numeric=force_y_numeric)
    return mojo_path, test


class H2OMojoWrapper:

    def __init__(self, mojo_path, column_names=None):
        self._mojo_path = mojo_path
        self._mojo_model = h2o.upload_mojo(mojo_path)
        self._column_names = column_names

    def __getstate__(self):
        return {
            "path": self._mojo_path,
            "colnames": self._column_names
        }

    def __setstate__(self, state):
        self._mojo_path = state.path
        self._mojo_model = h2o.upload_mojo(state.path)
        self._column_names = state.colnames

    def predict(self, arr):
        return self.predict_with_probabilities(arr)[0]

    def predict_with_probabilities(self, data):
        data_frame = H2OFrame(data, column_names=self._column_names)
        preds = self._mojo_model.predict(data_frame).as_data_frame(use_pandas=True)
        if len(preds.columns) == 1:
            return [preds.to_numpy()]
        else:
            return [
                preds.iloc[:, 0].to_numpy().astype(np.str),
                preds.iloc[:, 1:].to_numpy()
            ]


class TestH2OModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        h2o.init(port=54440)

    @classmethod
    def tearDownClass(cls):
        h2o.cluster().shutdown()

    def test_h2o_unsupported_algo(self):
        gbm = H2ORandomForestEstimator(ntrees=7, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        with self.assertRaises(ValueError) as err:
            _convert_mojo(mojo_path)
        self.assertRegexpMatches(err.exception.args[0], "not supported")

    def test_h2o_regressor_unsupported_dists(self):
        diabetes = load_diabetes()
        train, test = _train_test_split_as_frames(diabetes.data, diabetes.target)
        not_supported_dists = ["poisson", "gamma", "tweedie"]
        for d in not_supported_dists:
            gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5, distribution=d)
            mojo_path = _make_mojo(gbm, train)
            with self.assertRaises(ValueError) as err:
                _convert_mojo(mojo_path)
            self.assertRegexpMatches(err.exception.args[0], "not supported")

    def test_h2o_regressor(self):
        diabetes = load_diabetes()
        train, test = _train_test_split_as_frames(diabetes.data, diabetes.target)
        dists = ["auto", "gaussian", "huber", "laplace", "quantile"]
        for d in dists:
            gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5, distribution=d)
            mojo_path = _make_mojo(gbm, train)
            onnx_model = _convert_mojo(mojo_path)
            self.assertIsNot(onnx_model, None)
            dump_data_and_model(
                test,
                H2OMojoWrapper(mojo_path),
                onnx_model,
                basename="H2OReg-Dec4",
                allow_failure="StrictVersion("
                              "onnx.__version__)"
                              "< StrictVersion('1.3.0')",
            )

    def test_h2o_regressor_cat(self):
        y = "IsDepDelayed"
        train, test = _prepare_one_hot("airlines.csv", y, exclude_cols=["IsDepDelayed_REC"])
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model,
            basename="H2ORegCat-Dec4",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_multi_2class(self):
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5, distribution="multinomial")
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        with self.assertRaises(ValueError) as err:
            _convert_mojo(mojo_path)
        self.assertRegexpMatches(err.exception.args[0], "not supported")

    def test_h2o_classifier_bin_cat(self):
        y = "IsDepDelayed_REC"
        train, test = _prepare_one_hot("airlines.csv", y, exclude_cols=["IsDepDelayed"])
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model,
            basename="H2OClassBinCat",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_multi_cat(self):
        y = "fYear"
        train, test = _prepare_one_hot("airlines.csv", y)
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path = _make_mojo(gbm, train, y=train.columns.index(y))
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test.values.astype(np.float32),
            H2OMojoWrapper(mojo_path, list(test.columns)),
            onnx_model,
            basename="H2OClassMultiCat",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_bin_str(self):
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=True)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data,
            H2OMojoWrapper(mojo_path),
            onnx_model,
            basename="H2OClassBinStr",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_bin_int(self):
        gbm = H2OGradientBoostingEstimator(ntrees=8, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 2, is_str=False, force_y_numeric=True)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data,
            H2OMojoWrapper(mojo_path),
            onnx_model,
            basename="H2OClassBinInt",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_multi_str(self):
        gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 11, is_str=True)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data,
            H2OMojoWrapper(mojo_path),
            onnx_model,
            basename="H2OClassMultiStr",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_multi_int(self):
        gbm = H2OGradientBoostingEstimator(ntrees=9, max_depth=5)
        mojo_path, test_data = _train_classifier(gbm, 9, is_str=False)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test_data,
            H2OMojoWrapper(mojo_path),
            onnx_model,
            basename="H2OClassMultiBin",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )

    def test_h2o_classifier_multi_discrete_int_labels(self):
        iris = load_iris()
        x = iris.data[:, :2]
        y = iris.target
        y[y == 0] = 10
        y[y == 1] = 20
        y[y == 2] = -30
        train, test = _train_test_split_as_frames(x, y, is_str=False, is_classifier=True)
        gbm = H2OGradientBoostingEstimator(ntrees=7, max_depth=5)
        mojo_path = _make_mojo(gbm, train)
        onnx_model = _convert_mojo(mojo_path)
        self.assertIsNot(onnx_model, None)
        dump_data_and_model(
            test,
            H2OMojoWrapper(mojo_path),
            onnx_model,
            basename="H2OClassMultiDiscInt",
            allow_failure="StrictVersion("
                          "onnx.__version__)"
                          "< StrictVersion('1.3.0')",
        )


if __name__ == "__main__":
    unittest.main()
