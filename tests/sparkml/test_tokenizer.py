import pandas
import unittest
import sys
from pyspark.ml.feature import Tokenizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlTokenizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_tokenizer(self):
        data = self.spark.createDataFrame([("a b c",)], ["text"])
        model = Tokenizer(inputCol='text', outputCol='words')
        predicted = model.transform(data)

        model_onnx = convert_sparkml(model, 'Sparkml Tokenizer', [
            ('text', StringTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted_np = predicted.toPandas().words.apply(pandas.Series).values
        data_np = data.toPandas().text.values.reshape([1,1])
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlTokenizer")


if __name__ == "__main__":
    unittest.main()
