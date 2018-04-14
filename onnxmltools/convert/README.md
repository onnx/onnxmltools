# Conversion Framework

In ONNXMLTools, the conversion framework consists of several essential components listed below.

* Intermediate representation (defined in _topology.py)
    * Topology
    * Scope
    * Operator
    * Variable
* Parsers (defined in coreml/sklearn subdirectory's _parse.py)
    * Core ML parser
    * scikit-learn parser
* Complier (defined in _topology.py)
    * Graph optimization
    * Shape inference
    * Apply post-processing rules
    * Conduct basic checks
* Shape calculators (defined in coreml/sklearn's shape_calculators subdirectory)
    * Core ML shape calculators
    * scikit-learn shape calculators
* Converters (defined in coreml/sklearn's operator_converters subdirectory)
    * Core ML converters
    * scikit-learn converters
* Registration (defined in _registration.py)
    * shape calculator registration
    * converter registratoin

Notice that the design concept of this intermediate representation (IR) is to have some suitable data structures required to convert a computational graph from one format to another.

## Intermediate Representation (IR)

Again, we emphasize that this IR is not a formal language. It is just a collection of data stuctures and functions which hopefully can be shared by conversion between any two formats.

The `Topology` class is the top-level structure of a computational graph. A `Topology` object may contain several scopes and each scope can defines its own operators and variables. In `Topology`, we provide some functions for processing the whole graph. For example, `topological_operator_iterator` can traverse all the operators included in a topology like we're really executing the considered graph.

There are two major functionilities a `Scope` may provide. First, it includes a naming mechanism for variables/operators so that all variable/operator names are unique. Note that variables' naming mechnisms is independent from that of operators so that an operator and a variable can share the same name. Second, a `Scope` works like a container of operators and variables. Because two different `Scope` objects are essentially independent, we can use them in an recursive parsing algorithm to isolate components found at different stages.

`Variable` and `Operator` are the smallest objects in a computational graph. To encode the topological dependencies between operators, each `Operator` object has a input list and a output list. The two lists are python lists of `Variable` object. As you may expect, an operator computes its output(s) from its given input(s).

## Parsers

Parser is used to translate the considered raw model (e.g., a Core ML model) into our a `Topology` object. For Core ML, its parsing functions is defined in
`onnxmltools.convert.coreml._parse`. For scikit-learn's, please see `onnxmltools.convert.sklearn._parse`.

## Complier

Our complier is a collection of functions defined in `Topology` class. By calling `compile()` defined in `Topology`, those functions would be sequentially applied to the associated `Topology` object.

Our compliing procedure also aims at provide enough information so that at conversion stage, the conversions of all operators can happen in parallel.

### Shape inference

For each variable, its shape information, a list of integers and strings, is encoded in  its `type` field. To access the shape of a variable, `x`, you can do `x.type.shape`. Note that the only allowed string is `"None"`, which stands for a variable-length coordinate.

Our shape inference is a sequence of functions calls. Each call takes one `Operator` as the only input argument and calculate the `type` fields of all its output variables. Of course, a shape calculator is called only if all its input variables have been initialized or produced by some other operators.

The shape mapping from Core ML to our IR follows the following rules.

* Core ML's `[C, H, W]` is mapped to `[N, C, H, W]`
* Core ML's `[C]` is mapped to `[N, C]`
* Core ML's `[S, C]` is mapped to `[N, C]`
* Core ML's `[S, C, H, W]` is mapped to `[N, C, H, W]`
* Core ML's scalar shape (e.g., `Int64Type`) is mapped to `[1, 1]`

Core ML's batch size, `N-axis`, is ignored because it's is not related to graph structure. In fact, ONNX's batch size is rather equivalent to sequence axis in Core ML. By default, we use `N=1` for traditional machine learning models and `N='None'` for neural networks. To overwrite our default types, user can provide `initial_types` when calling `convert(...)` defined in `onnxmltools.convert.coreml.convert.py`. All Core ML's shape calculations are derived from [this document](https://apple.github.io/coremltools/coremlspecification/index.html) specifically for our type system.
Some more details about Core ML neural network operator can be found at this [page](https://github.com/apple/coremltools/blob/master/mlmodel/format/NeuralNetwork.proto)

For scikit-learn, user may need to specify the input types for their models. In general, we expect `[1, C]` if the input is feature vector.

If fortunately your model (e.g., Keras) includes shape information, you can register an empty function for all your operators.

## Converters

A converter is a function used to convert an `Operator` into some ONNX components. For example, to represent a Core ML LSTM, we may create several ONNX nodes and initializers.

Every converter has three input arguments, `scope`, `operator`, and `container`. The `scope` is a `Scope` object including some functions for declaring new operators and new variables. The `operator` is an `Operator` object, which is the major piece you need to convert. In addition to the two input and output lists, a `Operator` also contains the input model's operator and some other useful information. The last argument, `container`, is used to stored all ONNX objects created inside this converter. Note that all the ONNX objects stored in `container` will be passed to an ONNX `ModelProto` at the end of our conversion pipeline.

To invode converters in a topological order, call `convert_topology(...)` defined in _topology.py

## Registration

For each `Operator` type we want to support, one shape calculator and one converter function must be registrated. Detailed instructions can be found in `onnxmltools.convert.common._registration`.

## A Typical Model Conversion Procedure

A typical conversion process may include three steps.

* First, we translate the input model (commonly called raw model in our code) into our IR by calling a suitable parser.
* The second stage is compling. We may try to optimize the computational graph (i.e, a topology) to reduce redundant computations and then calculate the shapes of all existing variables. Also, post-processing rules and some basic checks are applied.
* Third, we may call a function to invode the conversions of all existing operators in a topologically order.
* Notice that all existing operators' shape calculators and converters should be registered.

This flow is implemented in both of our Core ML and scikit-learn conversions.
The main Core ML conver function is `onnxmltools.convert.coreml.convert`. For scikit-learn, see `onnxmltools.convert.sklearn.convert`.