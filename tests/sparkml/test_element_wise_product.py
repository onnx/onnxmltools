import sys
import unittest
import numpy
import pandas
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlElementwiseProduct(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_element_wise_product(self):
        data = self.spark.createDataFrame([(Vectors.dense([2.0, 1.0, 3.0]),)], ["features"])
        model = ElementwiseProduct(scalingVec=Vectors.dense([1.0, 2.0, 3.0]),
                                   inputCol="features", outputCol="eprod")
        feature_count = data.first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml ElementwiseProduct',
                                     [('features', FloatTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = [
            predicted.toPandas().eprod.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
            ]
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlElementwiseProduct")


if __name__ == "__main__":
    unittest.main()
