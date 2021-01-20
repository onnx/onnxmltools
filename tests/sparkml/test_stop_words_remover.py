# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from distutils.version import StrictVersion

import onnx
from pyspark.ml.feature import StopWordsRemover

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlStopWordsRemover(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.5'), 'Need Greater Opset 10')
    def test_stop_words_remover(self):
        data = self.spark.createDataFrame([(["a", "b", "c"],)], ["text"])
        model = StopWordsRemover(inputCol="text", outputCol="words", stopWords=["b"])

        feature_count = len(data.columns)
        model_onnx = convert_sparkml(model, 'Sparkml StopWordsRemover',
                                     [('text', StringTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().words.values
        data_np = data.toPandas().text.values
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlStopWordsRemover")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
