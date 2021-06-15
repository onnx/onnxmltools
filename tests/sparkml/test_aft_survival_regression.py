# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import pandas
import numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import AFTSurvivalRegression
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmAFTSurvivalRegression(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_aft_regression_survival(self):
        data = self.spark.createDataFrame([
            (1.0, Vectors.dense(1.0), 1.0),
            (1e-40, Vectors.sparse(1, [], []), 0.0)
        ], ["label", "features", "censor"])
        gbt = AFTSurvivalRegression()
        model = gbt.fit(data)
        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml AFTSurvivalRegression', [
            ('features', FloatTensorType([None, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlAFTSurvivalRegression")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
