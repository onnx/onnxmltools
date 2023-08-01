# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import packaging.version as pv
import numpy
import pandas
import onnx
from onnx.defs import onnx_opset_version
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors
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


class TestSparkmlVectorIndexer(SparkMlTestCase):
    @unittest.skipIf(
        True,
        reason=(
            "discrepency, unfound values are replaced by -1 by ONNX and 0 " "by spark."
        ),
    )
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.3"), "Need Greater Opset 9"
    )
    def test_model_vector_indexer_multi(self):
        vi = VectorIndexer(maxCategories=2, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([-1.0, 1.0, 3.1]),),
                (Vectors.dense([0.0, 5.0, 6.2]),),
                (Vectors.dense([0.0, 9.0, 3.1]),),
            ],
            ["a"],
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(
            model,
            "Sparkml VectorIndexer Multi",
            [("a", FloatTensorType([None, model.numFeatures]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.toPandas()
            .indexed.apply(lambda x: pandas.Series(x.toArray()))
            .values
        )
        data_np = (
            data.toPandas()
            .a.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlVectorIndexerMulti"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(["indexed"], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.3"), "Need Greater Opset 9"
    )
    def test_model_vector_indexer_single(self):
        vi = VectorIndexer(maxCategories=3, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([-1.0]),),
                (Vectors.dense([0.0]),),
                (Vectors.dense([0.0]),),
            ],
            ["a"],
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(
            model,
            "Sparkml VectorIndexer Single",
            [("a", FloatTensorType([None, model.numFeatures]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        expected = (
            predicted.toPandas()
            .indexed.apply(lambda x: pandas.Series(x.toArray()))
            .values
        )
        data_np = (
            data.toPandas()
            .a.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlVectorIndexerSingle"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(["indexed"], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
