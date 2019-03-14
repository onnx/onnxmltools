"""
Tests SparkML StringIndexer converter.
"""
import sys
import unittest

from pyspark.ml.feature import Imputer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model

## For some reason during the spark bring up and shutdown something happens causing Imputer
## tests to fail. For that you need to run each test here individually
## for now these will be commented out so as not to break the build
##      AttributeError: 'NoneType' object has no attribute 'setCallSite' on model.surrogateDF
##  Therefore we leave these tests out for now until a newere version of pyspark is availabe that address this issue
class TestSparkmlImputer(SparkMlTestCase):
    pass

    # @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    # def test_imputer(self):
    #     self._imputer_test_single()
    #     self._imputer_test_single()
    #
    # def _imputer_test_multi(self):
    #     import numpy
    #     data = self.spark.createDataFrame([
    #         (1.0, float("nan")),
    #         (2.0, float("nan")),
    #         (float("nan"), 3.0),
    #         (4.0, 4.0),
    #         (5.0, 5.0)
    #     ], ["a", "b"])
    #     imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
    #     model = imputer.fit(data)
    #
    #     # the input name should match the inputCols above
    #     model_onnx = convert_sparkml(model, 'Sparkml Imputer Multi Input', [
    #         ('a', FloatTensorType([1, 1])),
    #         ('b', FloatTensorType([1, 1]))
    #     ])
    #     self.assertTrue(model_onnx is not None)
    #
    #     # run the model
    #     predicted = model.transform(data)
    #     predicted_np = predicted.select("out_a", "out_b").toPandas().values.astype(numpy.float32)
    #     data_np = [ data.toPandas().values.astype(numpy.float32) ]
    #     dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlImputerMulti")
    #
    # def _imputer_test_single(self):
    #     import numpy
    #     data = self.spark.createDataFrame([
    #         (1.0, float("nan")),
    #         (2.0, float("nan")),
    #         (float("nan"), 3.0),
    #         (4.0, 4.0),
    #         (5.0, 5.0)
    #     ], ["a", "b"])
    #     imputer = Imputer(inputCols=["a"], outputCols=["out_a"])
    #     model = imputer.fit(data)
    #
    #     # the input name should match the inputCols above
    #     model_onnx = convert_sparkml(model, 'Sparkml Imputer', [
    #         ('a', FloatTensorType([1, 1]))
    #     ])
    #     self.assertTrue(model_onnx is not None)
    #
    #     # run the model
    #     predicted = model.transform(data)
    #     predicted_np = predicted.select("out_a").toPandas().values.astype(numpy.float32)
    #     data_np = data.toPandas().a.values.astype(numpy.float32)
    #     dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlImputerSingle")


if __name__ == "__main__":
    unittest.main()
