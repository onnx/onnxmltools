# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlPCA(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_model_polynomial_expansion(self):
        data = self.spark.createDataFrame([
            (Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)
        ], ["features"])
        pca = PCA(k=2, inputCol="features", outputCol="pca_features")
        model = pca.fit(data)

        # the input name should match that of what StringIndexer.inputCol
        feature_count = data.first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml PCA', [('features', FloatTensorType([None, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().pca_features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlPCA")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['pca_features'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
