# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import pandas
import numpy
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmMinHashLSH(SparkMlTestCase):

    @unittest.skipIf(True, reason="Discrepencies (Float -> Double?).")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_min_hash_lsh(self):
        data = self.spark.createDataFrame([
            (0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),),
            (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0]),),
            (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]),)
        ], ["id", "features"])
        mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
        model = mh.fit(data)

        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml MinHashLSH', [
            ('features', FloatTensorType([None, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data.limit(2))
        data_np = data.limit(2).toPandas().features.apply(
            lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().hashes.apply(
                lambda x: pandas.Series(x).map(
                    lambda y: y.values[0])).values.astype(numpy.float32)]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                 basename="SparkmlMinHashLSH")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['hashes'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
