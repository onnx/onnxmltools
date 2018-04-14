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
* Converters (defined in coreml/sklearn's shape_calculator subdirectory)
    * Core ML converters
    * scikit-learn converters
* Registration (defined in _registration.py)
    * shape calculator registration
    * converter registratoin

Notice that the design concept of this intermediate representation (IR) is to have some suitable data structures required to convert a computational graph from one format to another.

## Intermediate Representation (IR)

We emphasize that this IR is not a formal language. It is just a collection of data stuctures and functions which hopefully can be shared by conversion between any two formats.

The `Topology` class is the top-level structure of a computational graph. A `Topology` object may contain several scopes and each scope can defines its own operators and variables. In `Topology`, we provide some useful functions for processing the whole graph. For example, `topological_operator_iterator` can traverse all the operators included in a topology like we're really executing the graph.

There are two major functionilities a `Scope` may provide. First, it provides a naming mechanism for variables/operators so that all variable/operator names are unique. Note that variables' naming mechnisms is independent from that of operators so that an operator and a variable can share the same name. Second, a `Scope` works like a container of operators and variables. Because two different `Scope` object are essentially independent, we can use them in an recursive parsing algorithm to isolate components found at different stages.

`Variable` and `Operator` are the smallest objects in a computational graph. To encode the topological dependencies between operators, each `Operator` object has a input list and a output list. The two lists are python lists of `Variable` object. As you may expect, an operator computes its output(s) from its given input(s).

## Parsers

Parser is used to translate the considered raw model (e.g., a Core ML model) into our IR. For Core ML, its parsing functions is defined in
`onnxmltools.convert.coreml._parse`. For scikit-learn's parsing functions,we refer to `onnxmltools.convert.sklearn._parse`.  See their comments for details.

## Complier

Our complier is a collection of functions defined in `Topology` class. By calling `compile()` defined in `Topology`, those functions can be sequentially applied to the associated `Topology` object.

## Converters

A converter is a function used to convert an operator into some ONNX components. For example, to represent a Core ML LSTM, we may create several ONNX nodes and initializers.

Every converter has three input arguments, `scope`, `operator`, and `container`. The `scope` is a `Scope` object including some functions for declaring new operators and new variables. The `operator` is an `Operator` object, which is the major piece you need to convert. In addition to the two input and output lists, a `Operator` also contains the input model's operator and some other useful information.

To invode converters in a topological order, call `convert_topology()` defined in _topology.py

## Registration

See `onnxmltools.convert.common._registration` for details.

## A Typical Model Conversion Procedure

 A typical conversion process may include three steps.

* First, we translate the input model (commonly called raw model in our code) into our IR by calling a suitable parser.
* The second stage is compling. We may try to optimize the computational graph (i.e, a topology) to reduce redundant computations and then calculate the shapes of all existing variables. Also, post-processing rules and some basic checks are applied.
* Third, we may call a function to invode the conversions of all existing operators in a topologically order.
* Notice that all existing operators' shape calculators and converters should be registered.

See `onnxmltools.convert.coreml.convert` or `onnxmltools.convert.sklearn.convert` for a concrete example.