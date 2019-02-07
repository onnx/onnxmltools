"""
Tests SparkML OneHotEncoder converter.
"""
import unittest
from pyspark.ml.feature import OneHotEncoderEstimator
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_sparkml_model
from onnxmltools.utils.tests_spark_helper import start_spark,stop_spark


class TestSparkmlOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.spark = start_spark()


    def tearDown(self):
        stop_spark(self.spark)


    def test_model_onehot_encoder(self):
        import numpy
        encoder = OneHotEncoderEstimator(inputCols=['index'], outputCols=['indexVec'])
        data = self.spark.createDataFrame([(0.0,), (1.0,), (2.0,), (2.0,), (0.0,), (2.0,)], ['index'])
        model = encoder.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml OneHotEncoder', [('input', FloatTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        data_np = data.select("index").toPandas().values.astype(numpy.float32)
        predicted_np = predicted.select("indexVec").toPandas().indexVec.apply(lambda x: x.toArray().tolist()).values
        predicted_shifted = numpy.asarray([x + [0] if numpy.amax(x) == 1 else x + [1] for x in predicted_np])

        dump_data_and_sparkml_model(data_np, predicted_shifted, model, model_onnx,
                                basename="SparkmlOneHotEncoder")


if __name__ == "__main__":
    unittest.main()
