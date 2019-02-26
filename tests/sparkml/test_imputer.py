"""
Tests SparkML StringIndexer converter.
"""
import unittest

from pyspark.ml.feature import Imputer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_sparkml_model
from sparkml import SparkMlTestCase


class TestSparkmlImputer(SparkMlTestCase):

    def test_imputer_multi(self):
        import numpy
        data = self.spark.createDataFrame([
            (1.0, float("nan")),
            (2.0, float("nan")),
            (float("nan"), 3.0),
            (4.0, 4.0),
            (5.0, 5.0)
        ], ["a", "b"])
        imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
        model = imputer.fit(data)

        # the input name should match the inputCols above
        model_onnx = convert_sparkml(model, 'Sparkml Imputer Multi Input', [
            ('a', FloatTensorType([1, 1])),
            ('b', FloatTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.select("out_a", "out_b").toPandas().values.astype(numpy.float32)
        data_np = [ data.toPandas().values.astype(numpy.float32) ]
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlImputerMulti")

#
## The 2nd run of test always fails with error:
##      AttributeError: 'NoneType' object has no attribute 'setCallSite' on model.surrogateDF
##  Therefore we leave this out for now until a newere version of pyspark is availabe that address this issue
#
    # def test_imputer(self):
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
