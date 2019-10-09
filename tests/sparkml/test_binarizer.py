import sys
import unittest

from pyspark.ml.feature import Binarizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlBinarizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_binarizer(self):
        import numpy
        data = self.spark.createDataFrame([(0, 0.1), (1, 0.8), (2, 0.2) ], ["id", "feature"])
        model = Binarizer(inputCol='feature', outputCol='binarized')

        # the input name should match that of what StringIndexer.inputCol
        model_onnx = convert_sparkml(model, 'Sparkml Binarizer', [('feature', FloatTensorType([None, 1]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.select("binarized").toPandas().values.astype(numpy.float32)
        data_np = data.select('feature').toPandas().values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlBinarizer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['binarized'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

if __name__ == "__main__":
    unittest.main()
