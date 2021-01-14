# SPDX-License-Identifier: Apache-2.0

import inspect
import unittest
import sys
import os
import numpy
import pandas
from pyspark.ml import PipelineModel
from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml import buildInitialTypesSimple, buildInputDictSimple
from onnxmltools.utils.utils_backend_onnxruntime import run_with_runtime, _compare_expected
from tests.sparkml import SparkMlTestCase


class RPipeline(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_sparkml_r_pipeline(self):
        # add additional jar files before creating SparkSession
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "iris.csv")
        data = self.spark.read.format('csv') \
            .options(header='true', inferschema='true').load(input_path) \
            .drop('_index_')

        # read the model from disk
        pipeline_path = os.path.join(this_script_dir, "mlpmodel")
        model = PipelineModel.load(path=pipeline_path)

        # create Onnx model
        model_onnx = convert_sparkml(model, 'Sparkml R Pipeline', buildInitialTypesSimple(data), spark_session=self.spark)
        # save Onnx model for runtime usage
        if model_onnx is None: raise AssertionError("Failed to create the onnx model")
        model_path = os.path.join(this_script_dir, "tests_dump", "r_pipeline_model.onnx")
        with open(model_path, "wb") as f:
            f.write(model_onnx.SerializeToString())

        data_np = buildInputDictSimple(data)
        # run the model in Spark
        spark_prediction = model.transform(data)
        # run the model in onnx runtime
        output, session = run_with_runtime(data_np, model_path)

        # compare results
        expected = [
            spark_prediction.toPandas().label.values.astype(numpy.float32),
            spark_prediction.toPandas().prediction.values.astype(numpy.float32),
            spark_prediction.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(
                numpy.float32)
        ]
        _compare_expected(expected, output, session, model_path, decimal=5, onnx_shape=None)


if __name__ == "__main__":
    unittest.main()

