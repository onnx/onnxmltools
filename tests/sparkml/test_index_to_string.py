import sys
import unittest

import numpy
import pytest
from pyspark.ml.feature import IndexToString, StringIndexer
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import Int64TensorType
from onnxmltools.convert.sparkml.utils import SparkMlConversionError
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlIndexToString(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @pytest.mark.xfail(raises=SparkMlConversionError)
    def test_index_to_string_throws(self):
        original_data = self.spark.createDataFrame(
            [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
            ["id", "category"])
        string_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
        string_indexer_model = string_indexer.fit(original_data)
        data = string_indexer_model.transform(original_data)

        model = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
        # the input name should match that of what IndexToString.inputCol
        model_onnx = None
        with pytest.raises(SparkMlConversionError):
            model_onnx = convert_sparkml(model, 'Sparkml IndexToString', [('categoryIndex', Int64TensorType([None, 1]))])

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_index_to_string(self):
        original_data = self.spark.createDataFrame(
            [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
            ["id", "category"])
        string_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
        string_indexer_model = string_indexer.fit(original_data)
        data = string_indexer_model.transform(original_data)

        model = IndexToString(inputCol="categoryIndex", outputCol="originalCategory",
                              labels=['A', 'B', 'C'])
        # the input name should match that of what IndexToString.inputCol
        model_onnx = convert_sparkml(model, 'Sparkml IndexToString', [('categoryIndex', Int64TensorType([None, 1]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        expected = predicted.select("originalCategory").toPandas().values
        data_np = data.select('categoryIndex').toPandas().values.astype(numpy.int64)
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlIndexToString")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['originalCategory'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

if __name__ == "__main__":
    unittest.main()
