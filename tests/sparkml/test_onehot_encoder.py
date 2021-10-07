# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
from pyspark.ml.feature import OneHotEncoder
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlOneHotEncoder(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_model_onehot_encoder(self):
        encoder = OneHotEncoder(inputCols=['index'], outputCols=['indexVec'])
        data = self.spark.createDataFrame(
            [(0.0,), (1.0,), (2.0,), (2.0,), (0.0,), (2.0,)], ['index'])
        model = encoder.fit(data)
        model_onnx = convert_sparkml(
            model, 'Sparkml OneHotEncoder', [('index', FloatTensorType([None, 1]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        data_np = data.select("index").toPandas().values.astype(numpy.float32)
        predicted_np = predicted.select("indexVec").toPandas().indexVec.apply(
            lambda x: x.toArray().tolist()).values
        expected = numpy.asarray(
            [x + [0] if numpy.amax(x) == 1 else x + [1] for x in predicted_np])

        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlOneHotEncoder")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['indexVec'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
