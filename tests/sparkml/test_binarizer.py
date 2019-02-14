"""
Tests SparkML StringIndexer converter.
"""
import unittest

from pyspark.ml.feature import Binarizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_sparkml_model
from sparkml import SparkMlTestCase


class TestSparkmlBinarizer(SparkMlTestCase):
    def test_model_binarizer(self):
        import numpy
        data = self.spark.createDataFrame([(0, 0.1), (1, 0.8), (2, 0.2) ], ["id", "feature"])
        model = Binarizer(inputCol='feature', outputCol='binarized')

        # the input name should match that of what StringIndexer.inputCol
        model_onnx = convert_sparkml(model, 'Sparkml Binarizer', [('feature', FloatTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.select("binarized").toPandas().values.astype(numpy.float32)
        data_np = data.select('feature').toPandas().values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlBinarizer")


if __name__ == "__main__":
    unittest.main()
