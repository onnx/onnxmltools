import pandas
import sys
import unittest

import numpy
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlDCT(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_dct(self):
        data = self.spark.createDataFrame(
            [(Vectors.dense([5.0, 8.0, 6.0]),)],
            ["vec"])
        model = DCT(inverse=False, inputCol="vec", outputCol="resultVec")
        # the input name should match that of what inputCol
        feature_count = data.first()[0].size
        N = data.count()
        model_onnx = convert_sparkml(
            model, 'Sparkml DCT',
            [('vec', FloatTensorType([None, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().resultVec.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().vec.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = numpy.vstack([data_np, data_np])
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlDCT")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['resultVec'], data_np, onnx_model_path)
        expected = numpy.vstack([expected, expected])
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
