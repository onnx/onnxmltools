"""
Tests SparkML LogisticRegression converter.
"""
import unittest
import numpy
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.mllib.linalg import Vectors
from pyspark.sql.types import ArrayType, FloatType

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_sparkml_model
from onnxmltools.utils.tests_spark_helper import start_spark,stop_spark


class TestSparkmlLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.spark = start_spark()


    def tearDown(self):
        stop_spark(self.spark)


    def test_model_logistic_regression_binary_class(self):
        import inspect
        import os
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        original_data = self.spark.read.format("libsvm").load(input_path)
        #
        # truncate the features
        #
        self.spark.udf.register("truncateFeatures", lambda x: SparseVector(5, range(0,5), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("label", "truncateFeatures(features) as features")
        lr = LogisticRegression(maxIter=100, tol=0.0001)
        model = lr.fit(data)
        # print("sparkml coefficients", model.coefficients)
        # print("sparkml intercepts", model.intercept)
        model_onnx = convert_sparkml(model, 'sparkml logistic regression', [('input', FloatTensorType([1, model.numFeatures]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(data)
        #
        # sklearn equivalent
        #
        # import pandas
        # from sklearn import linear_model
        # from onnxmltools import convert_sklearn
        # self.spark.udf.register("sparseToArray", lambda x: x.toArray().tolist(),
        #                    ArrayType(elementType=FloatType(), containsNull=False))
        # np_y = data.select("label").toPandas().label.values.astype(numpy.float32)
        # np_x = data.selectExpr("sparseToArray(features) as features").toPandas().features.apply(pandas.Series).values.astype(numpy.float32)
        # sk_model = linear_model.LogisticRegression()
        # sk_model.fit(np_x, np_y)
        # sk_expected = [sk_model.predict(np_x), sk_model.predict_proba(np_x)]
        # sk_model_onnx = convert_sklearn(sk_model, 'logistic regression', [('input', FloatTensorType([1, model.numFeatures]))])
        # basename = "SklearnLogitisticRegressionBinary"
        # folder = "D:\\temp\\testdump"
        # dest = os.path.join(folder, basename + ".model.onnx")
        # with open(dest, "wb") as f:
        #     f.write(sk_model_onnx.SerializeToString())
        # print("sklearn coefficients", sk_model.coef_)
        # print("sklearn intercepts", sk_model.intercept_)
        # print("sk_expected labels: ", sk_expected[0])
        # print("sk_expected probabilities: ", sk_expected[1])

        # run the model
        import pandas
        predicted = model.transform(data)
        self.spark.udf.register("sparseToArray", lambda x: x.toArray().tolist(), ArrayType(elementType=FloatType(), containsNull=False))
        sql = get_conversion_sql(data)
        data_np = data.selectExpr(sql).toPandas().features.apply(pandas.Series).values.astype(numpy.float32)
        sql = get_conversion_sql(predicted)
        expected = [
            predicted.selectExpr(sql).select('prediction').toPandas().values.astype(numpy.float32),
            predicted.selectExpr(sql).select('probability').toPandas().probability.apply(pandas.Series).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlLogisticRegression")


def get_conversion_sql(df):
    cols = df.columns
    schema = df.schema
    sql = []
    for i in range(0, len(cols)):
        if isinstance(schema.fields[i].dataType, VectorUDT):
            sql.append("sparseToArray(" + cols[i] + ") as " + cols[i])
        else:
            sql.append(cols[i])
    return sql


if __name__ == "__main__":
    unittest.main()
