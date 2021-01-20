# SPDX-License-Identifier: Apache-2.0

from distutils.version import StrictVersion

import onnx
import pandas
import unittest
import sys
from pyspark.ml.feature import Tokenizer

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlTokenizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.5'), 'Need Greater Opset 10')
    def test_tokenizer(self):
        data = self.spark.createDataFrame([("a b c",)], ["text"])
        model = Tokenizer(inputCol='text', outputCol='words')
        predicted = model.transform(data)

        model_onnx = convert_sparkml(model, 'Sparkml Tokenizer', [
            ('text', StringTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        # run the model
        expected = predicted.toPandas().words.apply(pandas.Series).values
        data_np = data.toPandas().text.values.reshape([1, 1])
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlTokenizer")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
