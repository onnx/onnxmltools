# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
from pyspark.ml.feature import OneHotEncoder
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


class TestSparkmlOneHotEncoder(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_model_onehot_encoder_1(self):
        """
        Testing ONNX conversion for Spark OneHotEncoder when handleInvalid
        is set to "error" and dropLast set to False.
        """
        encoder = OneHotEncoder(
            inputCols=["index1", "index2"],
            outputCols=["index1Vec", "index2Vec"],
            handleInvalid="error",
            dropLast=False,
        )
        data = self.spark.createDataFrame(
            [
                (
                    0.0,
                    5.0,
                ),
                (
                    1.0,
                    4.0,
                ),
                (
                    2.0,
                    3.0,
                ),
                (
                    2.0,
                    2.0,
                ),
                (
                    0.0,
                    1.0,
                ),
                (
                    2.0,
                    0.0,
                ),
            ],
            ["index1", "index2"],
        )
        model = encoder.fit(data)
        model_onnx = convert_sparkml(
            model,
            "Sparkml OneHotEncoder",
            [
                ("index1", FloatTensorType([None, 1])),
                ("index2", FloatTensorType([None, 1])),
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        data_np = {
            "index1": data.select("index1").toPandas().values.astype(numpy.float32),
            "index2": data.select("index2").toPandas().values.astype(numpy.float32),
        }

        predicted_np_1 = (
            predicted.select("index1Vec")
            .toPandas()
            .index1Vec.apply(lambda x: x.toArray())
            .values
        )
        predicted_np_2 = (
            predicted.select("index2Vec")
            .toPandas()
            .index2Vec.apply(lambda x: x.toArray())
            .values
        )
        expected = {
            "index1Vec": numpy.asarray(predicted_np_1.tolist()),
            "index2Vec": numpy.asarray(predicted_np_2.tolist()),
        }

        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlOneHotEncoder"
        )
        onnx_model_path = paths[-1]

        output_names = ["index1Vec", "index2Vec"]
        output, output_shapes = run_onnx_model(output_names, data_np, onnx_model_path)
        actual_output = dict(zip(output_names, output))

        compare_results(expected["index1Vec"], actual_output["index1Vec"], decimal=5)
        compare_results(expected["index2Vec"], actual_output["index2Vec"], decimal=5)

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_model_onehot_encoder_2(self):
        """
        Testing ONNX conversion for Spark OneHotEncoder when handleInvalid
        is set to "keep" and dropLast set to True.
        """
        encoder = OneHotEncoder(
            inputCols=["index1", "index2"],
            outputCols=["index1Vec", "index2Vec"],
            handleInvalid="keep",
            dropLast=True,
        )
        data = self.spark.createDataFrame(
            [
                (
                    0.0,
                    5.0,
                ),
                (
                    1.0,
                    4.0,
                ),
                (
                    2.0,
                    3.0,
                ),
                (
                    2.0,
                    2.0,
                ),
                (
                    0.0,
                    1.0,
                ),
                (
                    2.0,
                    0.0,
                ),
            ],
            ["index1", "index2"],
        )
        test = self.spark.createDataFrame(
            [
                (
                    3.0,
                    7.0,
                )
            ],
            ["index1", "index2"],
        )  # invalid data

        model = encoder.fit(data)
        model_onnx = convert_sparkml(
            model,
            "Sparkml OneHotEncoder",
            [
                ("index1", FloatTensorType([None, 1])),
                ("index2", FloatTensorType([None, 1])),
            ],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(test)
        data_np = {
            "index1": test.select("index1").toPandas().values.astype(numpy.float32),
            "index2": test.select("index2").toPandas().values.astype(numpy.float32),
        }

        predicted_np_1 = (
            predicted.select("index1Vec")
            .toPandas()
            .index1Vec.apply(lambda x: x.toArray())
            .values
        )
        predicted_np_2 = (
            predicted.select("index2Vec")
            .toPandas()
            .index2Vec.apply(lambda x: x.toArray())
            .values
        )
        expected = {
            "index1Vec": numpy.asarray(predicted_np_1.tolist()),
            "index2Vec": numpy.asarray(predicted_np_2.tolist()),
        }

        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlOneHotEncoder"
        )
        onnx_model_path = paths[-1]

        output_names = ["index1Vec", "index2Vec"]
        output, output_shapes = run_onnx_model(output_names, data_np, onnx_model_path)
        actual_output = dict(zip(output_names, output))

        compare_results(expected["index1Vec"], actual_output["index1Vec"], decimal=5)
        compare_results(expected["index2Vec"], actual_output["index2Vec"], decimal=5)


if __name__ == "__main__":
    unittest.main()
