import sys
import unittest
from distutils.version import StrictVersion
import numpy
import pandas
import onnx
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlVectorIndexer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_model_vector_indexer_multi(self):
        vi = VectorIndexer(maxCategories=2, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame([
            (Vectors.dense([-1.0, 1.0, 3.1]),),
            (Vectors.dense([0.0, 5.0, 6.2]),),
            (Vectors.dense([0.0, 9.0, 3.1]),)],
            ["a"]
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml VectorIndexer Multi',  [
            ('a', FloatTensorType([None, model.numFeatures]))
        ], target_opset=9)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().indexed.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = data.toPandas().a.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlVectorIndexerMulti")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['indexed'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_model_vector_indexer_single(self):
        vi = VectorIndexer(maxCategories=3, inputCol="a", outputCol="indexed")
        data = self.spark.createDataFrame([
            (Vectors.dense([-1.0]),),
            (Vectors.dense([0.0]),),
            (Vectors.dense([0.0]),)],
            ["a"]
        )
        model = vi.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml VectorIndexer Single',  [
            ('a', FloatTensorType([None, model.numFeatures]))
        ], target_opset=9)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        expected = predicted.toPandas().indexed.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = data.toPandas().a.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlVectorIndexerSingle")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['indexed'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
