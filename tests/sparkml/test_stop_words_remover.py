"""
Tests SparkML NGram converter.
"""
import sys
import unittest

from pyspark import Row
from pyspark.ml.feature import NGram, StopWordsRemover

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlStopWordsRemover(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_stop_words_remover(self):
        data = self.spark.createDataFrame([(["a", "b", "c"],)], ["text"])
        model = StopWordsRemover(inputCol="text", outputCol="words", stopWords=["b"])

        feature_count = len(data.columns)
        model_onnx = convert_sparkml(model, 'Sparkml StopWordsRemover',
                                     [('text', StringTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().words.values
        data_np = data.toPandas().text.values
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlStopWordsRemover")


if __name__ == "__main__":
    unittest.main()
