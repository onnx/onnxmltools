import unittest
import sys
import numpy
from pyspark.ml.feature import Bucketizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlBucketizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_bucketizer(self):
        values = [(0.1,), (0.4,), (1.2,), (1.5,)]
        data = self.spark.createDataFrame(values, ["features"])
        model = Bucketizer(splits=[-float("inf"), 0.5, 1.4, float("inf")], inputCol="features", outputCol="buckets")

        feature_count = len(data.select('features').first())
        model_onnx = convert_sparkml(model, 'Sparkml Bucketizer', [
            ('features', FloatTensorType([None, feature_count]))
        ])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.setHandleInvalid("error").transform(data)
        expected = predicted.select("buckets").toPandas().values.astype(numpy.float32)
        data_np = [data.toPandas().values.astype(numpy.float32)]
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlBucketizer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['buckets'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
