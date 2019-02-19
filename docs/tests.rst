
===============================
Tests, Dependencies, Contribute
===============================


*onnxmltools* takes *onnx* as dependency.
If you choose to install `onnx` from its source code, 
you must set an environment variable `ONNX_ML=1` 
before installing `onnx` package.

This package uses `ONNX <https://github.com/onnx/onnx>`_, 
`NumPy <http://www.numpy.org/>`_, 
`ProtoBuf <https://developers.google.com/protocol-buffers/>`_.


*onnxmltools* converts models in ONNX format which
can be then used to compute predictions with the
backend of your choice. However, there exists a way
to automatically check every converter with
`onnxruntime <https://pypi.org/project/onnxruntime/>`_,
`onnxruntime <https://pypi.org/project/onnxruntime-gpu/>`_.

Test all existing converters
----------------------------

This process requires to clone the *onnxmltools* repository.
The following command runs all unit tests and generates
dumps of models, inputs, expected outputs and converted models
in folder ``TESTDUMP``.

```
python tests/main.py DUMP
```

It requires *onnxruntime*, *numpy* for most of the models,
*pandas* for transform related to text features,
*scipy* for sparse features. One test also requires
*keras* to test a custom operator. That means
*sklearn* or any machine learning library is requested.

Add a new converter
-------------------

Once the converter is implemented, a unit test is added
to test it works. At the end of the unit test, function
*dump_data_and_model* or any equivalent function must be called
to dump the expected output and the converted model.
Once these file are generated, a corresponding test must
be added in *tests_backend* to compute the prediction
with the runtime.
