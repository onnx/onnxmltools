import pandas
import sys
import unittest

import numpy
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlChiSqSelector(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_chi_sq_selector(self):
        data = self.spark.createDataFrame([
            (Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0),
            (Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0),
            (Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0)
        ], ["features", "label"])
        selector = ChiSqSelector(numTopFeatures=1, outputCol="selectedFeatures")
        model = selector.fit(data)
        print(model.selectedFeatures)

        # the input name should match that of what StringIndexer.inputCol
        feature_count = data.first()[0].size
        N = data.count()
        model_onnx = convert_sparkml(model, 'Sparkml ChiSqSelector', [('features', FloatTensorType([N, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().selectedFeatures.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlChiSqSelector")


if __name__ == "__main__":
    unittest.main()
