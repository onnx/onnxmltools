# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlPolynomialExpansion(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_model_polynomial_expansion(self):
        data = self.spark.createDataFrame([
            (Vectors.dense([1.2, 3.2, 1.3, -5.6]),),
            (Vectors.dense([4.3, -3.2, 5.7, 1.0]),),
            (Vectors.dense([0, 3.2, 4.7, -8.9]),)
        ], ["dense"])
        model = PolynomialExpansion(degree=2, inputCol="dense", outputCol="expanded")

        # the input name should match that of what StringIndexer.inputCol
        feature_count = data.first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml PolynomialExpansion', [('dense', FloatTensorType([None, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().expanded.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().dense.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlPolynomialExpansion")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['expanded'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
