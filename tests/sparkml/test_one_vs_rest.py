# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import packaging.version as pv
import inspect
import os
import onnx
import pandas
import numpy
from pyspark.ml.classification import LogisticRegression, OneVsRest
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


class TestSparkmOneVsRest(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    @unittest.skipIf(
        pv.Version(onnx.__version__) <= pv.Version("1.3"), "Need Greater Opset 9"
    )
    def test_one_vs_rest(self):
        this_script_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        input_path = os.path.join(
            this_script_dir, "data", "sample_multiclass_classification_data.txt"
        )
        data = self.spark.read.format("libsvm").load(input_path)
        lr = LogisticRegression(maxIter=100, tol=0.0001, regParam=0.01)
        ovr = OneVsRest(classifier=lr)
        model = ovr.fit(data)

        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(
            model,
            "Sparkml OneVsRest",
            [("features", FloatTensorType([None, feature_count]))],
            spark_session=self.spark,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        data_np = (
            data.toPandas()
            .features.apply(lambda x: pandas.Series(x.toArray()))
            .values.astype(numpy.float32)
        )
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
        ]
        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlOneVsRest"
        )
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(["prediction"], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
