import numpy
import pandas
import sys
import unittest

from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlVectorSlicer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_vector_slicer(self):
        data = self.spark.createDataFrame([
            (Vectors.dense([-2.0, 2.3, 0.0, 0.0, 1.0]), ),
            (Vectors.dense([0.0, 0.0, 0.0, 0.0, 0.0]), ),
            (Vectors.dense([0.6, -1.1, -3.0, 4.5, 3.3]), )], ["features"])
        model = VectorSlicer(inputCol="features", outputCol="sliced", indices=[1, 4])

        feature_count = data.first()[0].array.size
        model_onnx = convert_sparkml(model, 'Sparkml VectorSlicer',
                                     [('features', FloatTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)

        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().sliced.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx, basename="SparkmlVectorSlicer")


if __name__ == "__main__":
    unittest.main()
