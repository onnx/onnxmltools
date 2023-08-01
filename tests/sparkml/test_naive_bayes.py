# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark import Row
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    run_onnx_model,
    compare_results,
)
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlNaiveBayes(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_naive_bayes_bernoulli(self):
        data = self.spark.createDataFrame(
            [
                Row(label=0.0, weight=0.1, features=Vectors.dense([0.0, 0.0])),
                Row(label=0.0, weight=0.5, features=Vectors.dense([0.0, 1.0])),
                Row(label=1.0, weight=1.0, features=Vectors.dense([1.0, 0.0])),
            ]
        )
        nb = NaiveBayes(smoothing=1.0, modelType="bernoulli", weightCol="weight")
        model = nb.fit(data)
        feature_count = data.select("features").first()[0].size
        model_onnx = convert_sparkml(
            model,
            "Sparkml NaiveBayes Bernoulli",
            [("features", FloatTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas()
            .probability.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32),
        ]
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlNaiveBayesBernoulli"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["prediction", "probability"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_naive_bayes_multinomial(self):
        data = self.spark.createDataFrame(
            [
                Row(label=0.0, weight=0.1, features=Vectors.dense([0.0, 0.0])),
                Row(label=0.0, weight=0.5, features=Vectors.dense([0.0, 1.0])),
                Row(label=1.0, weight=1.0, features=Vectors.dense([1.0, 0.0])),
            ]
        )
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial", weightCol="weight")
        model = nb.fit(data)
        feature_count = data.select("features").first()[0].size
        model_onnx = convert_sparkml(
            model,
            "Sparkml NaiveBayes Multinomial",
            [("features", FloatTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas()
            .probability.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32),
        ]
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np,
            expected,
            model,
            model_onnx,
            basename="SparkmlNaiveBayesMultinomial",
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["prediction", "probability"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
