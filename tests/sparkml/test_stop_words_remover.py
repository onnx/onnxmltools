# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from distutils.version import StrictVersion
import numpy
import onnx
from pyspark.ml.feature import StopWordsRemover
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlStopWordsRemover(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.5'),
                     'Need Greater Opset 10')
    def test_stop_words_remover2(self):
        data = self.spark.createDataFrame([(["a", "b", "c"],)], ["text"])
        model = StopWordsRemover(inputCol="text", outputCol="words", stopWords=["b"])

        model_onnx = convert_sparkml(model, 'Sparkml StopWordsRemover',
                                     [('text', StringTensorType([None]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = numpy.array(predicted.toPandas().words.values[0])
        data_np = numpy.array(data.toPandas().text.values[0])
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlStopWordsRemover")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['words'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
