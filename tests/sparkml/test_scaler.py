# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import StandardScaler, MaxAbsScaler, MinMaxScaler
from pyspark.ml.linalg import Vectors
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


class TestSparkmlScaler(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_maxabs_scaler(self):
        data = self.spark.createDataFrame(
            [
                (
                    0,
                    Vectors.dense([1.0, 0.1, -1.0]),
                ),
                (
                    1,
                    Vectors.dense([2.0, 1.1, 1.0]),
                ),
                (
                    2,
                    Vectors.dense([3.0, 10.1, 3.0]),
                ),
            ],
            ["id", "features"],
        )
        scaler = MaxAbsScaler(inputCol="features", outputCol="scaled_features")
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(
            model,
            "Sparkml MaxAbsScaler",
            [("features", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.toPandas()
            .scaled_features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlMaxAbsScaler"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["scaled_features"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_minmax_scaler(self):
        data = self.spark.createDataFrame(
            [
                (
                    0,
                    Vectors.dense([1.0, 0.1, -1.0]),
                ),
                (
                    1,
                    Vectors.dense([2.0, 1.1, 1.0]),
                ),
                (
                    2,
                    Vectors.dense([3.0, 10.1, 3.0]),
                ),
            ],
            ["id", "features"],
        )
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(
            model,
            "Sparkml MinMaxScaler",
            [("features", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.toPandas()
            .scaled_features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlMinMaxScaler"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["scaled_features"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_standard_scaler(self):
        data = self.spark.createDataFrame(
            [
                (
                    0,
                    Vectors.dense([1.0, 0.1, -1.0]),
                ),
                (
                    1,
                    Vectors.dense([2.0, 1.1, 1.0]),
                ),
                (
                    2,
                    Vectors.dense([3.0, 10.1, 3.0]),
                ),
            ],
            ["id", "features"],
        )
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        model = scaler.fit(data)

        # the input names must match the inputCol(s) above
        model_onnx = convert_sparkml(
            model,
            "Sparkml StandardScaler",
            [("features", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.toPandas()
            .scaled_features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlStandardScaler"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(
            ["scaled_features"], data_np, onnx_model_path
        )
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
