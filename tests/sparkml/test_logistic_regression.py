"""
Tests SparkML LogisticRegression converter.
"""
import unittest
import numpy
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.sql.types import ArrayType, FloatType

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from sparkml import dump_data_and_sparkml_model
from sparkml import SparkMlTestCase


class TestSparkmlLogisticRegression(SparkMlTestCase):
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
        # the name of the input for Logistic Regression is 'features'
        model_onnx = convert_sparkml(model, 'sparkml logistic regression', [('features', FloatTensorType([1, model.numFeatures]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
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
