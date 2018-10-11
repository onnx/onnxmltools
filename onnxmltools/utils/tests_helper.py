#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------
import numpy
import pickle
import os
from ..convert.common.data_types import FloatTensorType


def dump_data_and_model(data, model, onnx=None, basename="model", folder="tests"):
    """
    Saves data with pickle, saves the model with pickle and onnx,
    runs and saves the predictions for the given model.
    This function is used to test a backend for onnx.
    
    :param data: any kind of data
    :param model: any model
    :param onnx: onnx model or None to use *onnxmltools* to convert it
        only if the model accepts one float vector
    :param basemodel: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param  folder: files are written in this folder,
        it is created if it does not exist
    :return: the four created files
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if hasattr(model, "predict"):
        prediction = model.predict(data)
    elif hasattr(model, "transform"):
        prediction = model.transform(data)
    else:
        raise TypeError("Model has not predict or transform method.")
    
    names = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(prediction, f)
    
    dest = os.path.join(folder, basename + ".data.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(data, f)
    
    dest = os.path.join(folder, basename + ".model.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(model, f)
        
    if onnx is None:
        array = numpy.array(data)
        onnx = convert_model(model, basename, 
                             [('input', FloatTensorType(list(array.shape)))])
    
    dest = os.path.join(folder, basename + ".model.onnx")
    names.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx.SerializeToString())
        
    return names


def convert_model(model, name, input_types):
    """
    Runs the appropriate conversion method.
    
    :param model: model, scikit-learn, keras, or coremltools object
    :return: onnx model
    """
    from sklearn.base import BaseEstimator
    if isinstance(model, BaseEstimator):
        from onnxmltools.convert import convert_sklearn
        return convert_sklearn(model, name, input_types)
    else:
        from keras.models import Model
        if isinstance(model, Model):
            from onnxmltools.convert import convert_keras
            return convert_keras(model, name, input_types)
        else:
            from onnxmltools.convert import convert_coreml
            return convert_coreml(model, name, input_types)
    
    