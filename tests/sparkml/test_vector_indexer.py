"""
Tests SparkML StringIndexer converter.
"""
import sys
import unittest
from distutils.version import StrictVersion

import onnx
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlVectorIndexer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_model_vector_indexer_multi(self):
        import numpy
        import pandas
        vi = VectorIndexer(maxCategories=2, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame([
            (Vectors.dense([-1.0, 1.0, 3.1]),),
            (Vectors.dense([0.0, 5.0, 6.2]),),
            (Vectors.dense([0.0, 9.0, 3.1]),)],
            ["a"]
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml VectorIndexer Multi',  [
            ('a', FloatTensorType([1, model.numFeatures]))
        ], target_opset=9)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().indexed.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = data.toPandas().a.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                    basename="SparkmlVectorIndexerMulti")

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_model_vector_indexer_single(self):
        import numpy
        import pandas
        vi = VectorIndexer(maxCategories=3, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame([
            (Vectors.dense([-1.0]),),
            (Vectors.dense([0.0]),),
            (Vectors.dense([0.0]),)],
            ["a"]
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml VectorIndexer Single',  [
            ('a', FloatTensorType([1, model.numFeatures]))
        ], target_opset=9)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.toPandas().indexed.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = data.toPandas().a.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                    basename="SparkmlVectorIndexerSingle")


if __name__ == "__main__":
    unittest.main()
