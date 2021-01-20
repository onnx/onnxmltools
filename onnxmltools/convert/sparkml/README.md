<!--- SPDX-License-Identifier: Apache-2.0 -->

# Spark ML to Onnx Model Conversion

There is prep work needed above and beyond calling the API. In short these steps are:

* providing the API with the types of Tensors being input to the Session.
* creating proper Tensors from the DataFrame you are going to use for prediction.
* taking the output Tensor(s) and converting it(them) back to a DataFrame if further processing is required.

## Instructions
For examples, please see the unit tests under `tests/sparkml`

1- Create a list of input types needed to be supplied to the `convert_sparkml()` call.
For simple cases you can use `buildInitialTypesSimple()` function in `convert/sparkml/utils.py`.
To use this function just pass your test DataFrame.

Otherwise, the conversion code requires a list of tuples with input names and their corresponding Tensor types, as shown below:
```python
initial_types = [
    ("label", StringTensorType([1, 1])),
    # (repeat for the required inputs)
]
```
Note that the input names are the same as columns names from your DataFrame and they must match the "inputCol(s)" values

you provided when you created your Pipeline.

2- Now you can create the ONNX model from your pipeline model like so:
```python
pipeline_model = pipeline.fit(training_data)
onnx_model = convert_sparkml(pipeline_model, 'My Sparkml Pipeline', initial_types)
```

3- (optional) You could save the ONNX model for future use or further examination by using the `SerializeToString()`
method of ONNX model

```python
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

4- Before running this model (e.g. using `onnxruntime`) you need to create a `dict` from the input data. This dictionay
 will have entries for each input name and its corresponding TensorData. For simple cases you could use the function
`buildInputDictSimple()` and pass your testing DataFrame to it. Otherwise, you need to create something like the following:

```python
input_data = {}
input_data['label'] = test_df.select('label').toPandas().values
# ... (repeat for all desired inputs)
```


5- (optional) You could save the converted input data for possible debugging or future reuse. See below:
```python
with open("input_data", "wb") as f:
    pickle.dump(input, f)
```

6- And finally run the newly converted ONNX model in the runtime:
```python
sess = onnxruntime.InferenceSession(onnx_model)
output = sess.run(None, input_data)

```
 This output may need further conversion back to a DataFrame.


## Known Issues

1. Overall invalid data handling is problematic and not implemented in most cases.
Make sure your data is clean.

2. OneHotEncoderEstimator must not drop the last bit: OneHotEncoderEstimator has an option
which you can use to make sure the last bit is included in the vector: `dropLast=False`

3. Use FloatTensorType for all numbers (instead of Int6t4Tensor or other variations)

4. Some conversions, such as the one for Word2Vec, can only handle batch size of 1 (one input row)

