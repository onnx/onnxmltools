import sys
import unittest

from pyspark.ml.feature import StandardScaler, MaxAbsScaler, MinMaxScaler
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlScaler(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_maxabs_scaler(self):
        import numpy
        import pandas
        data = self.spark.createDataFrame([
            (0, Vectors.dense([1.0, 0.1, -1.0]),),
            (1, Vectors.dense([2.0, 1.1, 1.0]),),
            (2, Vectors.dense([3.0, 10.1, 3.0]),)
        ], ["id", "features"])
        scaler = MaxAbsScaler(inputCol='features', outputCol='scaled_features')
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(model, 'Sparkml MaxAbsScaler', [('features', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().scaled_features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlMaxAbsScaler")

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_minmax_scaler(self):
        import numpy
        import pandas
        data = self.spark.createDataFrame([
            (0, Vectors.dense([1.0, 0.1, -1.0]),),
            (1, Vectors.dense([2.0, 1.1, 1.0]),),
            (2, Vectors.dense([3.0, 10.1, 3.0]),)
        ], ["id", "features"])
        scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(model, 'Sparkml MinMaxScaler', [('features', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().scaled_features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlMinMaxScaler")

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_standard_scaler(self):
        import numpy
        import pandas
        data = self.spark.createDataFrame([
            (0, Vectors.dense([1.0, 0.1, -1.0]),),
            (1, Vectors.dense([2.0, 1.1, 1.0]),),
            (2, Vectors.dense([3.0, 10.1, 3.0]),)
        ], ["id", "features"])
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(model, 'Sparkml StandardScaler', [('features', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().scaled_features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlStandardScaler")


if __name__ == "__main__":
    unittest.main()
