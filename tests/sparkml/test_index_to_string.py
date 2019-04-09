import sys
import unittest

import numpy
import pytest
from pyspark.ml.feature import IndexToString, StringIndexer
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import Int64TensorType
from onnxmltools.convert.sparkml.utils import SparkMlConversionError
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


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
            model_onnx = convert_sparkml(model, 'Sparkml IndexToString', [('categoryIndex', Int64TensorType([1, 1]))])

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
        model_onnx = convert_sparkml(model, 'Sparkml IndexToString', [('categoryIndex', Int64TensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.select("originalCategory").toPandas().values
        data_np = data.select('categoryIndex').toPandas().values.astype(numpy.int64)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                    basename="SparkmlIndexToString")


if __name__ == "__main__":
    unittest.main()
