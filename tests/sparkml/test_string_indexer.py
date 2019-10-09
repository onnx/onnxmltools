import sys
import unittest
from pyspark.ml.feature import StringIndexer
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlStringIndexer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_string_indexer(self):
        indexer = StringIndexer(inputCol='cat1', outputCol='cat1_index', handleInvalid='skip')
        data = self.spark.createDataFrame([("a",), ("b",), ("c",), ("a",), ("a",), ("c",)], ['cat1'])
        model = indexer.fit(data)
        # the input name should match that of what StringIndexer.inputCol
        model_onnx = convert_sparkml(model, 'Sparkml StringIndexer', [('cat1', StringTensorType([None, 1]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        expected = predicted.select("cat1_index").toPandas().values
        data_np = data.select('cat1').toPandas().values
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlStringIndexer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['cat1_index'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
