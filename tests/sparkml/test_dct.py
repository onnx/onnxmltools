import pandas
import sys
import unittest

import numpy
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from sparkml import dump_data_and_sparkml_model
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
        model_onnx = convert_sparkml(model, 'Sparkml DCT', [('vec', FloatTensorType([N, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().resultVec.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().vec.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlDCT")


if __name__ == "__main__":
    unittest.main()
