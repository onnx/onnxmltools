# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlNormalizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_normalizer_1(self):
        data = self.spark.createDataFrame([
          (0, Vectors.dense(1.0, 0.5, -1.0)),
          (1, Vectors.dense(2.0, 1.0, 1.0)),
          (2, Vectors.dense(4.0, 10.0, 2.0))
        ]).toDF("id", "features")
        model = Normalizer(inputCol='features', outputCol='norm_feature', p=1.0)

        model_onnx = convert_sparkml(model, 'Sparkml Normalizer', [('features', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().norm_feature.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlNormalizer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['norm_feature'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_normalizer_2(self):
        data = self.spark.createDataFrame([
          (0, Vectors.dense(1.0, 0.5, -1.0)),
          (1, Vectors.dense(2.0, 1.0, 1.0)),
          (2, Vectors.dense(4.0, 10.0, 2.0))
        ]).toDF("id", "features")
        model = Normalizer(inputCol='features', outputCol='norm_feature', p=2.0)

        model_onnx = convert_sparkml(model, 'Sparkml Normalizer', [('features', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().norm_feature.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlNormalizer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['norm_feature'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
