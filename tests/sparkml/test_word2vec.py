import pandas
import sys
import unittest

import numpy
from pyspark.ml.feature import Word2Vec
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlWord2Vec(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_word2vec(self):
        data = self.spark.createDataFrame([
            ("Hi I heard about Spark".split(" "), ),
            ("I wish Java could use case classes".split(" "), ),
            ("Logistic regression models are neat".split(" "), )
        ], ["text"])
        word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
        model = word2Vec.fit(data)
        vectors = model.getVectors()
        vectors.show(100, False)

        result = model.transform(data)
        result.show(100, False)

        # the input name should match that of inputCol
        feature_count = len(data.first()[0])
        model_onnx = convert_sparkml(model, 'Sparkml Word2Vec', [('text', StringTensorType([1, feature_count]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data.limit(1))
        predicted_np = predicted.toPandas().result.apply(lambda  x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        data_np = data.limit(1).toPandas().text.values
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                    basename="SparkmlWord2Vec")


if __name__ == "__main__":
    unittest.main()