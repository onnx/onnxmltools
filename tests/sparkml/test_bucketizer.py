import unittest
import sys
import numpy
from pyspark.ml.feature import Bucketizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlBucketizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_spark_bucketizer(self):
        values = [(0.1,), (0.4,), (1.2,), (1.5,)]
        data = self.spark.createDataFrame(values, ["features"])
        model = Bucketizer(splits=[-float("inf"), 0.5, 1.4, float("inf")], inputCol="features", outputCol="buckets")

        feature_count = len(data.select('features').first())
        model_onnx = convert_sparkml(model, 'Sparkml Bucketizer', [
            ('features', FloatTensorType([1, feature_count]))
        ])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.setHandleInvalid("error").transform(data)
        predicted_np = predicted.select("buckets").toPandas().values.astype(numpy.float32)
        data_np = [data.toPandas().values.astype(numpy.float32)]
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlBucketizer")


if __name__ == "__main__":
    unittest.main()
