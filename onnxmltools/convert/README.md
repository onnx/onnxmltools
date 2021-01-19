<!--- SPDX-License-Identifier: Apache-2.0 -->

# Conversion Framework

In ONNXMLTools, the conversion framework consists of several essential components listed below.

* Intermediate representation (defined in _topology.py)
    * Topology
    * Scope
    * Operator
    * Variable
    * Type (defined in data_types.py)
* Containers
    * RawModelContainer
        * CoremlModelContainer
        * SklearnModelContainer
        * KerasModelContainer
* Parsers (defined in coreml/sklearn/keras subdirectory's _parse.py)
    * Core ML parser
    * scikit-learn parser
    * Keras parser
* Compiler (defined in _topology.py)
    * Graph optimization
    * Shape inference
    * Apply post-processing rules
    * Conduct basic checks
* Shape calculators (defined in coreml/sklearn/keras's shape_calculators subdirectory)
    * Core ML shape calculators
    * scikit-learn shape calculators
    * Keras shape calculators
* Converters (defined in coreml/sklearn/keras's operator_converters subdirectory)
    * Core ML converters
    * scikit-learn converters
    * Keras converters
* Registration (defined in _registration.py)
    * shape calculator registration
    * converter registration

Notice that the design concept of this intermediate representation (IR) is to have some suitable data structures required to convert a computational graph from one format to another.

## Intermediate Representation (IR)

Again, we emphasize that this IR is not a formal language. It is just a collection of data structures and functions which hopefully can be shared for conversion between any two formats.

The `Topology` class is the top-level structure of a computational graph. A `Topology` object may contain several scopes and each scope can define its own operators and variables. That is, the hierarchy relation of these objects may be `Topology` > `Scope` > `Operator` = `Variable`.
In `Topology`, we provide some functions for processing the whole graph. For example, `topological_operator_iterator` can traverse all the operators included in a topology like we're really executing the considered graph.

There are two major functions a `Scope` may provide. First, it includes a naming mechanism for variables/operators so that all variable/operator names are unique. Note that variables' naming mechanism is independent from that of operators so that an operator and a variable can share the same name. Second, a `Scope` works like a container of operators and variables. Because two different `Scope` objects are essentially independent, we can use them in a recursive parsing algorithm to isolate components found at different stages.

`Variable` and `Operator` are the smallest objects in a computational graph. To encode the topological dependencies between operators, each `Operator` object has an input list and an output list. The two lists are python lists of `Variable` objects. As you may expect, an operator computes its output(s) from its given input(s). One important attribute of a `Variable` object is its `type` field (i.e., a member variable in C++). Allowed `type` values such as `FloatTensorType` and `Int64TensorType` are defined in `onnxconverter_common.data_types`. Shape information is also included in `type`. To access the shape of a variable, `x`, you can do `x.type.shape`, which returns a list of integers and strings. Note that the only allowed string is `'None'`, which stands for a variable-length coordinate.

## Containers

Our framework relies on two different types of containers.

The first one is `RawModelContainer` and its derived classes. These objects are used to store the raw model (the one you want to convert into ONNX) and its input and output names. Let's provide an example explaining what are those names. If a CoreML model has input `feature_vector` and output `class_probabilities`, calling the property `input_names`/`output_names` of `RawModelContainer` should yield `['feature_vector']`/`['class_probabilities']`. These names basically defines the roots and leaves of your original computational graph. If we forget assigning an input name, it may cause some unreachable sub-graph which will be pruned in our compiling phase.

## Parsers

A parser is used to translate the considered raw model (e.g., a Core ML model) into a `Topology` object. For Core ML, its parsing algorithm is defined in
`onnxmltools.convert.coreml._parse`. Keras parse is implemented in `onnxmltools.convert.keras._parse`.

## Compiler

Our compiler is a collection of functions defined in `Topology` class. By calling `compile()` defined in `Topology`, those functions would be sequentially applied to the associated `Topology` object.

The compiling process generates all necessary information for every single operator's conversion. Once a topology is compiled, the subsequent conversions can happen independently and in parallel.

### Shape Inference

Our shape inference is a sequence of function calls. Each call takes one `Operator` as the only input argument and calculate the `type` (including `type`'s `shape` field) fields of all its output variables. Of course, a shape calculator is called only if all its input variables have been initialized or produced by some other operators.

The shape mapping from Core ML to our IR obeys the following rules.

* Core ML's `[C, H, W]` is mapped to `[N, C, H, W]`
* Core ML's `[C]` is mapped to `[N, C]`
* Core ML's `[S, C]` is mapped to `[N, C]`
* Core ML's `[S, C, H, W]` is mapped to `[N, C, H, W]`
* Core ML's scalar shape (e.g., `Int64Type`) is mapped to `[1, 1]` because they are all kind of features of an example.

Notice that the compiler can overwrite those rules at some stages like shape inference. An example is the label shape of a classifier. One may expect that its shape is `[1, 1].` Nevertheless, our shape inference may change it to `[1]`. The major reason is that the current definition of ZipMap, the operator used to generate the predicted probabilities, does not support batch size greater than one.

Core ML's batch size, `N-axis`, is ignored because it is not related to graph structures. In fact, ONNX's batch size is rather equivalent to sequence axis in Core ML. By default, we use `N=1` for traditional machine learning models and `N='None'` for neural networks. To overwrite our default types, user can provide `initial_types` when calling `convert(...)` defined in `onnxmltools.convert.coreml.convert.py`. All Core ML's shape calculations are derived from [this document](https://apple.github.io/coremltools/coremlspecification/index.html) specifically for our type system.
Some more details about Core ML neural network operator can be found at this [page](https://github.com/apple/coremltools/blob/master/mlmodel/format/NeuralNetwork.proto)

For scikit-learn, user may need to specify the input types for their models. In general, we expect `[1, C]` if the input is feature vector.

If fortunately your model (e.g., Keras) includes shape information, you can register an empty function for all your operators.

## Converters

A converter is a function used to convert an `Operator` into some ONNX components. For example, to represent a Core ML LSTM, we may create several ONNX nodes and initializers.

Every converter has three input arguments, `scope`, `operator`, and `container`. The `scope` is a `Scope` object including some functions for declaring new operators and new variables. The `operator` is an `Operator` object, which is the major piece you need to convert. A `operator` contains input and output lists which specify what `operator` should consume and produce, respectively. The computation (i.e., generating the outputs from the inputs) conducted by an `operator` is described by its `raw_operator` field (e.g., a scikit-learn random forest classifier) and the converter may follow `raw_operator` to create some ONNX objects (e.g., `NodeProto`). The last argument, `container`, is used to create and store all ONNX objects created inside this converter. Note that all the ONNX objects stored in `container` will be passed to an ONNX `ModelProto` at the end of our conversion pipeline.

The ONNX objects created by a converter can be viewed as a sub-graph where roots and leaves are specified by `operator`'s input and output lists and `nodes` are the ONNX operators used to simulate `raw_operator`'s behavior. The converter needs to make sure that those nodes in that sub-graph are connected correctly by properly assigning input and output names. To create ONNX operator names and ONNX variable names when composing a sub-graph (we don't need to create `Variable` and `Operator` at all because having names is enough for connecting ONNX nodes), the naming functions in `scope` should be called.

To invoke converters in a topological order, call `convert_topology(...)` defined in _topology.py

## Registration

For each `Operator` type we want to support, one shape calculator and one converter function must be registrated. Detailed instructions can be found in `onnxconverter_common.registration`.

## A Typical Model Conversion Procedure

A typical conversion process may include three steps.
First, we translate the input model (commonly called a raw model in our code) into our IR by calling a suitable parser. Each operator in the input model may be mapped to one or several `Operator` objects. Also, we may declare `Variable` objects to capture that operator's input(s) and output(s).
The second stage is compiling. We may try to optimize the computational graph (i.e, a topology) and then calculate the shapes of all existing variables. Also, post-processing rules and some basic checks may be applied.
Third, we may call a function to invoke the conversions of all existing operators in a topological order.
This procedure is implemented in both of our Core ML and scikit-learn conversions. The main Core ML conver function is `onnxmltools.convert.coreml.convert`. For scikit-learn, see `onnxmltools.convert.sklearn.convert`. Notice that each operator existing in your raw model must have one shape calculators and one converter registered (it can be an empty function if your parser is able to extract those information directly from the input model).
