import sys
import unittest
from pyspark.ml.feature import OneHotEncoderEstimator
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlOneHotEncoder(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_onehot_encoder(self):
        import numpy
        encoder = OneHotEncoderEstimator(inputCols=['index'], outputCols=['indexVec'])
        data = self.spark.createDataFrame([(0.0,), (1.0,), (2.0,), (2.0,), (0.0,), (2.0,)], ['index'])
        model = encoder.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml OneHotEncoder', [('index', FloatTensorType([1, 1]))])
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
