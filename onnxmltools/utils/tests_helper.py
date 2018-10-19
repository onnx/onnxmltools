#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------
import numpy
import pickle
import os
from ..convert.common.data_types import FloatTensorType


def dump_data_and_model(data, model, onnx=None, basename="model", folder="tests",
                        inputs=None):
    """
    Saves data with pickle, saves the model with pickle and *onnx*,
    runs and saves the predictions for the given model.
    This function is used to test a backend (runtime) for *onnx*.

    :param data: any kind of data
    :param model: any model
    :param onnx: *onnx* model or *None* to use *onnxmltools* to convert it
        only if the model accepts one float vector
    :param basemodel: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param folder: files are written in this folder,
        it is created if it does not exist
    :param inputs: standard type or specific one if specified, only used is
        parameter *onnx* is None
    :return: the four created files

    Some convention for the name,
    *Bin* for a binary classifier, *Mcl* for a multiclass
    classifier, *Reg* for a regressor, *MRg* for a multi-regressor.
    The name can contain some flags. Expected outputs refer to the
    outputs computed with the original library, computed outputs
    refer to the outputs computed with a ONNX runtime.
    
    * ``-CannotLoad``: the model can be converted but the runtime cannot load it
    * ``-Dec3``: compares expected and computed outputs up to 3 decimals (5 by default)
    * ``-Dec4``: compares expected and computed outputs up to 4 decimals (5 by default)    
    * ``-Disc``: the runtime fails due to discrepencies
    * ``-Mism``: the runtime fails due to a dimension mismatch between expected
    * ``-NoProb``: The original models computed probabilites for two classes *size=(N, 2)*
      but the runtime produces a vector of size *N*, the test will compare the second column
      to the column
    * ``-OneOff``: the ONNX runtime cannot computed the prediction for several inputs,
      it must be called for each of them
      and computed output.
    * ``-Out0``: only compares the first output on both sides
    * ``-Reshape``: merges all outputs into one single vector and resizes it before comparing
    * ``-SkipDim1``: before comparing expected and computed output,
      arrays with a shape like *(2, 1, 2)* becomes *(2, 2)*
    
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if hasattr(model, "predict"):
        if hasattr(model, "predict_proba"):
            # Classifier
            prediction = [model.predict(data), model.predict_proba(data)]
        elif hasattr(model, "decision_function"):
            # Classifier without probabilities
            prediction = [model.predict(data), model.decision_function(data)]
        else:
            # Regressor
            prediction = [model.predict(data)]
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
        if inputs is None:
            inputs = [('input', FloatTensorType(list(array.shape)))]
        onnx = convert_model(model, basename, inputs)
    
    dest = os.path.join(folder, basename + ".model.onnx")
    names.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx.SerializeToString())
        
    return names


def convert_model(model, name, input_types):
    """
    Runs the appropriate conversion method.
    
    :param model: model, *scikit-learn*, *keras*, or *coremltools* object
    :return: *onnx* model
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
    
    