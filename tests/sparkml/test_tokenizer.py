# SPDX-License-Identifier: Apache-2.0

import packaging.version as pv
import unittest
import sys
import onnx
import pandas
from pyspark.ml.feature import Tokenizer
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    run_onnx_model,
    compare_results,
)
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlTokenizer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.5"), "Need Greater Opset 10"
    )
    def test_tokenizer(self):
        data = self.spark.createDataFrame([("a b c",)], ["text"])
        model = Tokenizer(inputCol="text", outputCol="words")
        predicted = model.transform(data)

        model_onnx = convert_sparkml(
            model,
            "Sparkml Tokenizer",
            [("text", StringTensorType([None]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # run the model
        expected = predicted.toPandas().words.apply(pandas.Series).values
        data_np = data.toPandas().text.values.reshape([-1])
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlTokenizer"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(["words"], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
