# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

class SparkMLTree(dict):
    pass


def sparkml_tree_dataset_to_sklearn(tree_df, is_classifier):
    import numpy
    feature = []
    threshold = []
    tree_pandas = tree_df.toPandas()
    children_left = tree_pandas.leftChild.values.tolist()
    children_right = tree_pandas.rightChild.values.tolist()
    value = tree_pandas.impurityStats.values.tolist() if is_classifier else tree_pandas.prediction.values.tolist()
    split = tree_pandas.split.apply(tuple).values
    for item in split:
        feature.append(item[0])
        threshold.append(item[1][0] if len(item[1]) >= 1 else -1.0)
    tree = SparkMLTree()
    tree.children_left = children_left
    tree.children_right = children_right
    tree.value = numpy.asarray(value, dtype=numpy.float32)
    tree.feature = feature
    tree.threshold = threshold
    tree.node_count = tree_df.count()
    return tree


def save_read_sparkml_model_data(spark, model):
    import tempfile
    import os
    import time
    tdir = tempfile.tempdir
    if tdir is None:
        tdir = spark.util.Utils.createTempDir().getAbsolutePath()
    if tdir is None:
        raise FileNotFoundError(
            "Unable to create a temporary directory for model '{}'"
            ".".format(type(model).__name__))
    path = os.path.join(tdir, type(model).__name__ + "_" + str(time.time()))
    model.write().overwrite().save(path)
    df = spark.read.parquet(os.path.join(path, 'data'))
    return df
