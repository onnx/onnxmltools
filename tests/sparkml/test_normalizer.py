import sys
import unittest

from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlNormalizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_normalizer_1(self):
        import numpy
        import pandas
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
        predicted_np = predicted.toPandas().norm_feature.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlNormalizer")

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_normalizer_2(self):
        import numpy
        import pandas
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
        predicted_np = predicted.toPandas().norm_feature.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlNormalizer")


if __name__ == "__main__":
    unittest.main()
