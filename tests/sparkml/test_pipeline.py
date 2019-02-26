"""
Tests SparkML Pipeline converter.
"""
import unittest

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlPipeline(SparkMlTestCase):

    def test_model_pipeline_4_stage(self):
        import inspect
        import os
        import numpy
        import pandas
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "AdultCensusIncomeOriginal.csv")
        full_data = self.spark.read.format('csv')\
            .options(header='true', inferschema='true').load(input_path)
        cols = ['workclass', 'education', 'marital_status']
        training_data, test_data = full_data.select('income', *cols).limit(1000).randomSplit([0.9, 0.1],seed=1)

        stages = []
        for col in cols:
            stages.append(StringIndexer(inputCol=col, outputCol=col+'_index', handleInvalid='skip'))
            stages.append(OneHotEncoderEstimator(inputCols=[col+'_index'], outputCols=[col+'_vec'], dropLast=False))

        stages.append(VectorAssembler(inputCols=[c+'_vec' for c in cols], outputCol='features'))
        stages.append(StringIndexer(inputCol='income', outputCol='label', handleInvalid='skip'))
        stages.append(LogisticRegression(maxIter=100, tol=0.0001))
        pipeline = Pipeline(stages=stages)

        model = pipeline.fit(training_data)
        model_onnx = convert_sparkml(model, 'Sparkml Pipeline', [
            ('income', StringTensorType([1, 1])),
            ('workclass', StringTensorType([1, 1])),
            ('education', StringTensorType([1, 1])),
            ('marital_status', StringTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(test_data)
        data_np = {
            'income': test_data.select('income').toPandas().values,
            'workclass': test_data.select('workclass').toPandas().values,
            'education': test_data.select('education').toPandas().values,
            'marital_status': test_data.select('marital_status').toPandas().values
        }
        expected = [
            predicted.toPandas().label.values.astype(numpy.float32),
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                basename="SparkmlPipeline_4Stage")


    def test_model_pipeline_3_stage(self):
        import inspect
        import os
        import numpy
        import pandas
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "AdultCensusIncomeOriginal.csv")
        full_data = self.spark.read.format('csv')\
            .options(header='true', inferschema='true').load(input_path)
        cols = ['workclass', 'education', 'marital_status']
        training_data, test_data = full_data.select(*cols).limit(1000).randomSplit([0.9, 0.1], seed=1)

        stages = []
        for col in cols:
            stages.append(StringIndexer(inputCol=col, outputCol=col+'_index', handleInvalid='skip'))
            # we need the dropLast option otherwise when assembled together (below)
            # we won't be able to expand the features without difficulties
            stages.append(OneHotEncoderEstimator(inputCols=[col+'_index'], outputCols=[col+'_vec'], dropLast=False))

        stages.append(VectorAssembler(inputCols=[c+'_vec' for c in cols], outputCol='features'))
        pipeline = Pipeline(stages=stages)

        model = pipeline.fit(training_data)
        model_onnx = convert_sparkml(model, 'Sparkml Pipeline', [
            ('workclass', StringTensorType([1, 1])),
            ('education', StringTensorType([1, 1])),
            ('marital_status', StringTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(test_data)
        data_np = {
            'workclass': test_data.select('workclass').toPandas().values,
            'education': test_data.select('education').toPandas().values,
            'marital_status': test_data.select('marital_status').toPandas().values
        }
        predicted_np = predicted.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values
        dump_data_and_sparkml_model(data_np, predicted_np, model, model_onnx,
                                basename="SparkmlPipeline_3Stage")


    def test_model_pipeline_2_stage(self):
        import inspect
        import os
        import numpy
        import pandas
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "AdultCensusIncomeOriginal.csv")
        full_data = self.spark.read.format('csv')\
            .options(header='true', inferschema='true').load(input_path)
        cols = ['workclass', 'education', 'marital_status']
        training_data, test_data = full_data.select(*cols).limit(1000).randomSplit([0.9, 0.1], seed=1)

        stages = []
        for col in cols:
            stages.append(StringIndexer(inputCol=col, outputCol=col+'_index', handleInvalid='skip'))
            stages.append(OneHotEncoderEstimator(inputCols=[col+'_index'], outputCols=[col+'_vec']))

        pipeline = Pipeline(stages=stages)

        model = pipeline.fit(training_data)
        model_onnx = convert_sparkml(model, 'Sparkml Pipeline', [
            ('workclass', StringTensorType([1, 1])),
            ('education', StringTensorType([1, 1])),
            ('marital_status', StringTensorType([1, 1]))
        ])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        predicted = model.transform(test_data)
        data_np = {
            'workclass': test_data.select('workclass').toPandas().values,
            'education': test_data.select('education').toPandas().values,
            'marital_status': test_data.select('marital_status').toPandas().values
        }
        predicted_np = [
            predicted.toPandas().workclass_vec.apply(lambda x: pandas.Series(x.toArray())).values,
            predicted.toPandas().education_vec.apply(lambda x: pandas.Series(x.toArray())).values,
            predicted.toPandas().marital_status_vec.apply(lambda x: pandas.Series(x.toArray())).values
            ]
        expected = [numpy.asarray([expand_one_hot_vec(x) for x in row]) for row in predicted_np]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                basename="SparkmlPipeline_2Stage")

def expand_one_hot_vec(v):
    import numpy
    if numpy.amax(v) == 1:
        return v.tolist() + [0]
    else:
        return v.tolist() + [1]


if __name__ == "__main__":
    unittest.main()
