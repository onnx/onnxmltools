# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import packaging.version as pv
import onnx
import pandas
import numpy

try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    compare_results,
    run_onnx_model,
)
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmDecisionTreeClassifierBig(SparkMlTestCase):
    # @unittest.skipIf(True, reason="Mismatched input dimensions.")
    @ignore_warnings(category=(ResourceWarning, DeprecationWarning))
    @unittest.skipIf(sys.platform == "win32", reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.3"), "Need Greater Opset 9"
    )
    def test_tree_pipeline_category(self):
        ok = {1, 6, 10, 56, 78, 34}
        ok2 = {1, 7, 8, 23, 35, 45}

        def f_label(x):
            if x[1] in ok or x[0] in ok:
                return 1
            if x[1] in ok or x[0] in ok2:
                return 2
            return 0

        features = numpy.random.randint(0, high=50, size=(1000, 2)) % 100
        labels = numpy.array([f_label(row) for row in features])
        n_features = features.shape[1]
        df = pandas.DataFrame(features)
        features_names = [f"c{i}" for i in range(df.shape[1])]
        df.columns = features_names
        cat = set(s for s in df["c0"])
        df["c0"] = pandas.Categorical(
            list(map(lambda s: int(s), df["c0"])), categories=cat, ordered=False
        )
        cat = set(s for s in df["c1"])
        df["c1"] = pandas.Categorical(
            list(map(lambda s: int(s), df["c1"])), categories=cat, ordered=False
        )
        df["label"] = labels

        sparkDF = self.spark.createDataFrame(df)

        data = sparkDF  # self.spark.read.csv(input_path, header=True, inferSchema=True)
        va = VectorAssembler(inputCols=features_names, outputCol="features")
        va_df = va.transform(data)
        va_df = va_df.select(["features", "label"])

        dt = DecisionTreeClassifier(
            labelCol="label", featuresCol="features", maxDepth=3, maxBins=50
        )
        model = dt.fit(va_df)
        # print(model.toDebugString)
        model_onnx = convert_sparkml(
            model,
            "Sparkml Decision Tree Binary Class",
            [("features", FloatTensorType([None, n_features]))],
            spark_session=self.spark,
            target_opset=TARGET_OPSET,
        )
        data_np = (
            va_df.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        predicted = model.transform(va_df)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas()
            .probability.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32),
        ]
        paths = save_data_models(
            data_np,
            expected,
            model,
            model_onnx,
            basename="SparkmlDecisionTreeBinaryClassCategory",
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["prediction", "probability"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    import logging

    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    unittest.main()
