# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    run_onnx_model,
    compare_results,
)
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlVectorAssembler(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_model_vector_assembler(self):
        col_names = ["a", "b"]
        model = VectorAssembler(inputCols=col_names, outputCol="features")
        data = self.spark.createDataFrame(
            [(1.0, Vectors.dense([1.0, 2.0, 3.0]))], col_names
        )
        model_onnx = convert_sparkml(
            model,
            "Sparkml VectorAssembler",
            [("a", FloatTensorType([None, 1])), ("b", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.select("features")
            .toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values
        )
        data_np = {
            "a": data.select("a").toPandas().values.astype(numpy.float32),
            "b": data.select("b")
            .toPandas()["b"]
            .apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32),
        }
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlVectorAssembler"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(["features"], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)
        assert output_shapes == [[None, 4]]


if __name__ == "__main__":
    unittest.main()
