"""
Tests SparkML StringIndexer converter.
"""
import sys
import unittest
from pyspark.ml.feature import VectorAssembler
from onnxmltools import convert_sparkml
from onnxtk.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlStringIndexer(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_vector_assembler(self):
        import numpy
        import pandas
        col_names = ["a", "b", "c"]
        model = VectorAssembler(inputCols=col_names, outputCol='features')
        data = self.spark.createDataFrame([(1., 0., 3.)], col_names)
        model_onnx = convert_sparkml(model, 'Sparkml VectorAssembler',  [
            ('a', FloatTensorType([1, 1])),
            ('b', FloatTensorType([1, 1])),
            ('c', FloatTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        predicted_np = predicted.select("features").toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values
        data_np = {
            'a': data.select('a').toPandas().values.astype(numpy.float32),
            'b': data.select('b').toPandas().values.astype(numpy.float32),
            'c': data.select('c').toPandas().values.astype(numpy.float32)
        }
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                    basename="SparkmlVectorAssembler")


if __name__ == "__main__":
    unittest.main()
