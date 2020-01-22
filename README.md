
<p align="center"><img width="40%" src="docs/ONNXMLTools_logo_main.png" /></p>

| Linux | Windows |
|-------|---------|
| [![Build Status](https://dev.azure.com/onnxmltools/onnxmltools/_apis/build/status/onnxmltools-linux-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/onnxmltools/_build/latest?definitionId=3?branchName=master)| [![Build Status](https://dev.azure.com/onnxmltools/onnxmltools/_apis/build/status/onnxmltools-win32-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/onnxmltools/_build/latest?definitionId=3?branchName=master)|

# Introduction 
ONNXMLTools enables you to convert models from different machine learning toolkits into [ONNX](https://onnx.ai). Currently the following toolkits are supported:
* Keras (a wrapper of [keras2onnx converter](https://github.com/onnx/keras-onnx/))
* Tensorflow (a wrapper of [tf2onnx converter](https://github.com/onnx/tensorflow-onnx/))
* scikit-learn (a wrapper of [skl2onnx converter](https://github.com/onnx/sklearn-onnx/))
* Apple Core ML
* Spark ML (experimental)
* LightGBM
* libsvm
* XGBoost
* H2O

## Install
You can install latest release of ONNXMLTools from [PyPi](https://pypi.org/project/onnxmltools/):
```
pip install onnxmltools
```
or install from source:
```
pip install git+https://github.com/microsoft/onnxconverter-common
pip install git+https://github.com/onnx/onnxmltools
```
If you choose to install `onnxmltools` from its source code, you must set the environment variable `ONNX_ML=1` before installing the `onnx` package. 

## Dependencies
This package relies on ONNX, NumPy, and ProtoBuf. If you are converting a model from scikit-learn, Core ML, Keras, LightGBM, SparkML, XGBoost, H2O or LibSVM, you will need an environment with the respective package installed from the list below:
1. scikit-learn
2. CoreMLTools
3. Keras (version 2.0.8 or higher) with the corresponding Tensorflow version
4. LightGBM (scikit-learn interface)
5. SparkML
6. XGBoost (scikit-learn interface)
7. libsvm
8. H2O

ONNXMLTools has been tested with Python **2.7**, **3.5**, **3.6**, and **3.7**.  
  `Note: some wrapped converters may not support python 2.x anymore.`

# Examples
If you want the converted ONNX model to be compatible with a certain ONNX version, please specify the target_opset parameter upon invoking the convert function. The following Keras model conversion example demonstrates this below. You can identify the mapping from ONNX Operator Sets (referred to as opsets) to ONNX releases in the [versioning documentation](https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions). 

## Keras to ONNX Conversion
Next, we show an example of converting a Keras model into an ONNX model with `target_opset=7`, which corresponds to ONNX release version 1.2.

```python
import onnxmltools
from keras.layers import Input, Dense, Add
from keras.models import Model

# N: batch size, C: sub-model input dimension, D: final model's input dimension
N, C, D = 2, 3, 3

# Define a sub-model, it will become a part of our final model
sub_input1 = Input(shape=(C,))
sub_mapped1 = Dense(D)(sub_input1)
sub_model1 = Model(inputs=sub_input1, outputs=sub_mapped1)

# Define another sub-model, it will become a part of our final model
sub_input2 = Input(shape=(C,))
sub_mapped2 = Dense(D)(sub_input2)
sub_model2 = Model(inputs=sub_input2, outputs=sub_mapped2)

# Define a model built upon the previous two sub-models
input1 = Input(shape=(D,))
input2 = Input(shape=(D,))
mapped1_2 = sub_model1(input1)
mapped2_2 = sub_model2(input2)
sub_sum = Add()([mapped1_2, mapped2_2])
keras_model = Model(inputs=[input1, input2], output=sub_sum)

# Convert it! The target_opset parameter is optional.
onnx_model = onnxmltools.convert_keras(keras_model, target_opset=7) 
```

## CoreML to ONNX Conversion
Here is a simple code snippet to convert a Core ML model into an ONNX model.

```python
import onnxmltools
import coremltools

# Load a Core ML model
coreml_model = coremltools.utils.load_spec('example.mlmodel')

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, 'Example Model')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'example.onnx')
```

## Spark ML to ONNX Conversion (experimental)
Please refer to the following documents:
 * [Conversion Framework](onnxmltools/convert/README.md)
 * [Spark ML to ONNX Model Conversion](onnxmltools/convert/sparkml/README.md)

## H2O to ONNX Conversion
Below is a code snippet to convert a H2O MOJO model into an ONNX model. The only pre-requisity is to have a MOJO model saved on the local file-system.

```python
import onnxmltools

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_h2o('/path/to/h2o/gbm_mojo.zip')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'h2o_gbm.onnx')
```

# Testing model converters

*onnxmltools* converts models into the ONNX format which
can be then used to compute predictions with the
backend of your choice. 

## Checking the operator set version of your converted ONNX model

You can check the operator set of your converted ONNX model using [Netron](https://github.com/lutzroeder/Netron), a viewer for Neural Network models. Alternatively, you could identify your converted model's opset version through the following line of code.

```
opset_version = onnx_model.opset_import[0].version
```

If the result from checking your ONNX model's opset is smaller than the `target_opset` number you specified in the onnxmltools.convert function, be assured that this is likely intended behavior. The ONNXMLTools converter works by converting each operator to the ONNX format individually and finding the corresponding opset version that it was most recently updated in. Once all of the operators are converted, the resultant ONNX model has the maximal opset version of all of its operators.

To illustrate this concretely, let's consider a model with two operators, Abs and Add. As of December 2018, [Abs](https://github.com/onnx/onnx/blob/master/docs/Operators.md#abs) was most recently updated in opset 6, and [Add](https://github.com/onnx/onnx/blob/master/docs/Operators.md#add) was most recently updated in opset 7. Therefore, the converted ONNX model's opset will always be 7, even if you request `target_opset=8`. The converter behavior was defined this way to ensure backwards compatibility. 

Documentation for the [ONNX Model format](https://github.com/onnx/onnx) and more examples for converting models from different frameworks can be found in the [ONNX tutorials](https://github.com/onnx/tutorials) repository. 

## Test all existing converters

There exists a way
to automatically check every converter with
[onnxruntime](https://pypi.org/project/onnxruntime/) or
[onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/).
This process requires the user to clone the *onnxmltools* repository.
The following command runs all unit tests and generates
dumps of models, inputs, expected outputs and converted models
in folder ``TESTDUMP``.

```
python tests/main.py DUMP
```

It requires *onnxruntime*, *numpy* for most models,
*pandas* for transforms related to text features, and
*scipy* for sparse features. One test also requires
*keras* to test a custom operator. That means
*sklearn* or any machine learning library is requested.

## Add a new converter

Once the converter is implemented, a unit test is added
to confirm that it works. At the end of the unit test, function
*dump_data_and_model* or any equivalent function must be called
to dump the expected output and the converted model.
Once these file are generated, a corresponding test must
be added in *tests_backend* to compute the prediction
with the runtime.

# License
[MIT License](LICENSE)
