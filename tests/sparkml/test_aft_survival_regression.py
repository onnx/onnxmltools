import sys
import unittest

import pandas
import numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import AFTSurvivalRegression

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmAFTSurvivalRegression(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_aft_regression_survival(self):
        data = self.spark.createDataFrame([
            (1.0, Vectors.dense(1.0), 1.0),
            (1e-40, Vectors.sparse(1, [], []), 0.0)
        ], ["label", "features", "censor"])
        gbt = AFTSurvivalRegression()
        model = gbt.fit(data)
        feature_count = data.first()[1].size
        model_onnx = convert_sparkml(model, 'Sparkml AFTSurvivalRegression', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlAFTSurvivalRegression")


if __name__ == "__main__":
    unittest.main()
