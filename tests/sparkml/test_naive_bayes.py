"""
Tests SparkML NGram converter.
"""
import pandas
import sys
import unittest

import numpy
from pyspark import Row
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlNaiveBayes(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_naive_bayes_bernoulli(self):
        data = self.spark.createDataFrame([
            Row(label=0.0, weight=0.1, features=Vectors.dense([0.0, 0.0])),
            Row(label=0.0, weight=0.5, features=Vectors.dense([0.0, 1.0])),
            Row(label=1.0, weight=1.0, features=Vectors.dense([1.0, 0.0]))])
        nb = NaiveBayes(smoothing=1.0, modelType="bernoulli", weightCol="weight")
        model = nb.fit(data)
        feature_count = data.select('features').first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml NaiveBayes Bernoulli',
                                     [('features', FloatTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted.show(20, False)
        predicted_np = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
            ]
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlNaiveBayesBernoulli")

    def test_naive_bayes_multinomial(self):
        data = self.spark.createDataFrame([
            Row(label=0.0, weight=0.1, features=Vectors.dense([0.0, 0.0])),
            Row(label=0.0, weight=0.5, features=Vectors.dense([0.0, 1.0])),
            Row(label=1.0, weight=1.0, features=Vectors.dense([1.0, 0.0]))])
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial", weightCol="weight")
        model = nb.fit(data)
        feature_count = data.select('features').first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml NaiveBayes Multinomial',
                                     [('features', FloatTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted.show(20, False)
        predicted_np = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
            ]
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlNaiveBayesMultinomial")

if __name__ == "__main__":
    unittest.main()
