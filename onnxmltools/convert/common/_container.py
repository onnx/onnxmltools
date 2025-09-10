# SPDX-License-Identifier: Apache-2.0

import abc
import re
from onnx import helper
from .data_types import TensorType
from ._registration import get_shape_calculator


class ModelContainer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_initializer(self, name, onnx_type, shape, content):
        """
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        """
        return

    @abc.abstractmethod
    def add_node(self, op_type, inputs, outputs, op_domain="", op_version=1, **attrs):
        """
        Add a NodeProto into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        """
        return


class OperatorBase:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        pass

    @property
    @abc.abstractmethod
    def input_full_names(self):
        """
        Return all input variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def output_full_names(self):
        """
        Return all outpu variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def original_operator(self):
        """
        Return the original operator/layer
        """
        pass


class ScopeBase:
    __metaclass__ = abc.ABCMeta

    pass


class ModelComponentContainer(ModelContainer):
    """
    In the conversion phase, this class is used to collect all materials required to build an ONNX GraphProto, which is
    encapsulated in a ONNX ModelProto.
    """

    def __init__(self, target_opset):
        """
        :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
        :param targeted_onnx: A string, for example, '1.1.2' and '1.2'.
        """
        # Inputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.inputs = []
        # Outputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.outputs = []
        # ONNX tensors (type: TensorProto). They are initializers of ONNX GraphProto.
        self.initializers = []
        # Intermediate variables in ONNX computational graph. They are ValueInfoProto in ONNX.
        self.value_info = []
        # ONNX nodes (type: NodeProto) used to define computation structure
        self.nodes = []
        # ONNX operators' domain-version pair set. They will be added into opset_import field in the final ONNX model.
        self.node_domain_version_pair_sets = set()
        # The targeted ONNX operator set (referred to as opset) that matches the ONNX version.
        self.target_opset = target_opset
        self.enable_optimizer = False

    def _make_value_info(self, variable):
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        if variable.type.doc_string:
            value_info.doc_string = variable.type.doc_string
        return value_info

    def add_input(self, variable):
        """
        Add our Variable object defined _parser.py into the the input list of the final ONNX model

        :param variable: The Variable object to be added
        """
        self.inputs.append(self._make_value_info(variable))

    def add_output(self, variable):
        """
        Add our Variable object defined _parser.py into the the output list of the final ONNX model

        :param variable: The Variable object to be added
        """
        self.outputs.append(self._make_value_info(variable))

    def add_initializer(self, name, onnx_type, shape, content):
        """
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        """
        if any(d is None for d in shape):
            raise ValueError("Shape of initializer cannot contain None")
        tensor = helper.make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)

    def add_value_info(self, variable):
        self.value_info.append(self._make_value_info(variable))

    def add_node(self, op_type, inputs, outputs, op_domain="", op_version=1, **attrs):
        """
        Add a NodeProto into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        """

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)) or not all(
            isinstance(s, str) for s in inputs
        ):
            type_list = ",".join(list(str(type(s)) for s in inputs))
            raise ValueError("Inputs must be a list of string but get [%s]" % type_list)
        if not isinstance(outputs, (list, tuple)) or not all(
            isinstance(s, str) for s in outputs
        ):
            type_list = ",".join(list(str(type(s)) for s in outputs))
            raise ValueError(
                "Outputs must be a list of string but get [%s]" % type_list
            )
        for k, v in attrs.items():
            if v is None:
                raise ValueError(
                    "Failed to create ONNX node. Undefined attribute pair (%s, %s) found"
                    % (k, v)
                )

        node = helper.make_node(op_type, inputs, outputs, **attrs)
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)


class RawModelContainer(object):
    """
    This container is the carrier of the model we want to convert. It provides an abstract layer so that our parsing
    framework can work with models generated by different tools.
    """

    def __init__(self, raw_model):
        self._raw_model = raw_model

    @property
    def raw_model(self):
        return self._raw_model

    @property
    def input_names(self):
        """
        This function should return a list of strings. Each string corresponds to an input variable name.
        :return: a list of string
        """
        raise NotImplementedError()

    @property
    def output_names(self):
        """
        This function should return a list of strings. Each string corresponds to an output variable name.
        :return: a list of string
        """
        raise NotImplementedError()


class CommonSklearnModelContainer(RawModelContainer):

    def __init__(self, sklearn_model):
        super(CommonSklearnModelContainer, self).__init__(sklearn_model)
        # Scikit-learn models have no input and output specified, so we create them and store them in this container.
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        # The order of adding variables matters. The final model's input names are sequentially added as this list
        if variable not in self._inputs:
            self._inputs.append(variable)

    def add_output(self, variable):
        # The order of adding variables matters. The final model's output names are sequentially added as this list
        if variable not in self._outputs:
            self._outputs.append(variable)


class LightGbmModelContainer(CommonSklearnModelContainer):
    pass


class XGBoostModelContainer(CommonSklearnModelContainer):
    pass


class H2OModelContainer(CommonSklearnModelContainer):
    pass


class SparkmlModelContainer(RawModelContainer):
    def __init__(self, sparkml_model):
        super(SparkmlModelContainer, self).__init__(sparkml_model)
        # Sparkml models have no input and output specified,
        # so we create them and store them in this container.
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        # The order of adding variables matters. The final model's
        # input names are sequentially added as this list
        if variable not in self._inputs:
            self._inputs.append(variable)

    def add_output(self, variable):
        # The order of adding variables matters.
        # The final model's output names are sequentially added as this list
        if variable not in self._outputs:
            self._outputs.append(variable)


class CoremlModelContainer(RawModelContainer):
    def __init__(self, coreml_model):
        super(CoremlModelContainer, self).__init__(coreml_model)

    @property
    def input_names(self):
        return [str(var.name) for var in self.raw_model.description.input]

    @property
    def output_names(self):
        return [str(var.name) for var in self.raw_model.description.output]


class LibSvmModelContainer(CommonSklearnModelContainer):
    pass


class Variable:

    def __init__(self, raw_name, onnx_name, scope, type=None):
        """
        :param raw_name: A string indicating the variable's name in the original model. Usually, it's the seed string
        used to created its ONNX name (i.e., the field onnx_name below).
        :param onnx_name: A string indicating the variable's name in the converted model
        :param scope: A string. It's the name of the scope where this variable is declared
        :param type: A type object defined in onnxmltools.convert.common.data_types.py; e.g., FloatTensorType
        """
        self.raw_name = raw_name  #
        self.onnx_name = onnx_name  #
        self.scope = scope
        self.type = type
        # The following fields are bool variables used in parsing and compiling stages
        self.is_fed = None
        self.is_root = None
        self.is_leaf = None
        self.is_abandoned = False
        self.raw_params = None

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.onnx_name

    def __str__(self):
        if self.raw_name != self.onnx_name:
            return "Var(name='{0}', onnx='{1}', type={2})".format(
                self.raw_name, self.onnx_name, self.type
            )
        else:
            return "Var(name='{0}', type={1})".format(self.raw_name, self.type)


class Operator(OperatorBase):

    def __init__(self, onnx_name, scope, type, raw_operator, target_opset):
        """
        :param onnx_name: A unique ID, which is a string
        :param scope: The name of the scope where this operator is declared. It's a string.
        :param type: A object which uniquely characterizes the type of this operator. For example, it can be a string,
        pooling, if this operator is associated with a CoreML pooling layer.
        :param raw_operator: The original operator which defines this operator; for example, a scikit-learn Imputer and
        a CoreML Normalizer.
        :param target_opset: The target opset number for the converted model.
        """
        self.onnx_name = onnx_name  # operator name in the converted model
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = []
        self.outputs = []
        self.is_evaluated = None
        self.is_abandoned = False
        self.target_opset = target_opset

    @property
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        return self.onnx_name

    @property
    def input_full_names(self):
        """
        Return all input variables' names
        """
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        """
        Return all output variables' names
        """
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        """
        Return the original operator/layer
        """
        return self.raw_operator

    def infer_types(self):
        # Invoke a core inference function
        get_shape_calculator(self.type)(self)


class Scope(ScopeBase):

    def __init__(
        self,
        name,
        parent_scopes=None,
        variable_name_set=None,
        operator_name_set=None,
        target_opset=None,
    ):
        """
        :param name:  A string, the unique ID of this scope in a Topology object
        :param parent_scopes: A list of Scope objects. The last element should be the direct parent scope (i.e., where
        this scope is declared).
        :param variable_name_set: A set of strings serving as the name pool of variables
        :param operator_name_set: A set of strings serving as the name pool of operators
        :param target_opset: The target opset number for the converted model.
        """
        self.name = name
        self.parent_scopes = parent_scopes if parent_scopes else list()
        self.onnx_variable_names = (
            variable_name_set if variable_name_set is not None else set()
        )
        self.onnx_operator_names = (
            operator_name_set if operator_name_set is not None else set()
        )
        self.target_opset = target_opset

        # An one-to-many map from raw variable name to ONNX variable names. It looks like
        #   (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ..., onnx_nameN])
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope. (key, value) = (onnx_name, variable)
        self.variables = {}

        # A map of local operators defined in this scope. (key, value) = (onnx_name, operator)
        self.operators = {}

    def get_onnx_variable_name(self, seed):
        """
        Retrieve the variable ID of the given seed or create one if it is the first time of seeing this seed
        """
        if seed in self.variable_name_mapping:
            return self.variable_name_mapping[seed][-1]
        else:
            return self.get_unique_variable_name(seed)

    def get_unique_variable_name(self, seed):
        """
        Create a unique variable ID based on the given seed
        """
        return Topology._generate_unique_name(seed, self.onnx_variable_names)

    def get_unique_operator_name(self, seed):
        """
        Create a unique operator ID based on the given seed
        """
        return Topology._generate_unique_name(seed, self.onnx_operator_names)

    def find_sink_variables(self):
        """
        Find sink variables in this scope
        """
        # First we assume all variables are sinks
        is_sink = {name: True for name in self.variables.keys()}
        # Then, we remove those variables which are inputs of some operators
        for operator in self.operators.values():
            for variable in operator.inputs:
                is_sink[variable.onnx_name] = False
        return [variable for name, variable in self.variables.items() if is_sink[name]]

    def declare_local_variable(self, raw_name, type=None, prepend=False):
        """
        This function may create a new variable in this scope. If raw_name has been used to create other variables,
        the new variable will hide all other variables created using raw_name.
        """
        # Get unique ID for the new variable
        onnx_name = self.get_unique_variable_name(raw_name)

        # Create the variable
        variable = Variable(raw_name, onnx_name, self.name, type)
        self.variables[onnx_name] = variable

        if raw_name in self.variable_name_mapping:
            # Hide existing variables with the same raw_name
            if not prepend:
                self.variable_name_mapping[raw_name].append(onnx_name)
            else:
                self.variable_name_mapping[raw_name].insert(0, onnx_name)
        else:
            self.variable_name_mapping[raw_name] = [onnx_name]
        return variable

    def get_local_variable_or_declare_one(self, raw_name, type=None):
        """
        This function will first check if raw_name has been used to create some variables. If yes, the latest one
        named in self.variable_name_mapping[raw_name] will be returned. Otherwise, a new variable will be created and
        then returned.
        """
        onnx_name = self.get_onnx_variable_name(raw_name)
        if onnx_name in self.variables:
            return self.variables[onnx_name]
        else:
            variable = Variable(raw_name, onnx_name, self.name, type)
            self.variables[onnx_name] = variable
            if raw_name in self.variable_name_mapping:
                self.variable_name_mapping[raw_name].append(onnx_name)
            else:
                self.variable_name_mapping[raw_name] = [onnx_name]
            return variable

    def declare_local_operator(self, type, raw_model=None):
        """
        This function is used to declare new local operator.
        """
        onnx_name = self.get_unique_operator_name(str(type))
        operator = Operator(onnx_name, self.name, type, raw_model, self.target_opset)
        self.operators[onnx_name] = operator
        return operator

    def delete_local_operator(self, onnx_name):
        """
        Remove the operator whose onnx_name is the input onnx_name
        """
        if onnx_name not in self.onnx_operator_names or onnx_name not in self.operators:
            raise RuntimeError("The operator to be removed not found")
        self.onnx_operator_names.discard(onnx_name)
        del self.operators[onnx_name]

    def delete_local_variable(self, onnx_name):
        """
        Remove the variable whose onnx_name is the input onnx_name
        """
        if onnx_name not in self.onnx_variable_names or onnx_name not in self.variables:
            raise RuntimeError("The variable to be removed not found")
        self.onnx_variable_names.discard(onnx_name)
        raw_name = self.variables[onnx_name].raw_name
        self.variable_name_mapping[raw_name].remove(onnx_name)
        del self.variables[onnx_name]


class Topology:

    def __init__(
        self,
        model,
        default_batch_size=1,
        initial_types=None,
        reserved_variable_names=None,
        reserved_operator_names=None,
        target_opset=None,
        custom_conversion_functions=None,
        custom_shape_calculators=None,
        metadata_props=None,
    ):
        """
        Initialize a Topology object, which is an intermediate representation of a computational graph.

        :param model: RawModelContainer object or one of its derived classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and array types from CoreML. It's usually 1 or 'None'.
        :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
        name and a type defined in data_types.py.
        :param reserved_variable_names: A set of strings which are not allowed to be used as a variable name
        :param reserved_operator_names: A set of strings which are not allowed to be used as a operator name
        :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
        """
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.variable_name_set = (
            reserved_variable_names if reserved_variable_names is not None else set()
        )
        self.operator_name_set = (
            reserved_operator_names if reserved_operator_names is not None else set()
        )
        self.initial_types = initial_types if initial_types else list()
        self.metadata_props = metadata_props if metadata_props else dict()
        self.default_batch_size = default_batch_size
        self.target_opset = target_opset
        self.custom_conversion_functions = (
            custom_conversion_functions if custom_conversion_functions else {}
        )
        self.custom_shape_calculators = (
            custom_shape_calculators if custom_shape_calculators else {}
        )

        # This attribute is used in optimizing the graph structure. If root_names is not empty, only the variables
        # specified will be treated as the roots (i.e., set is_fed to True in the beginning of a graph evaluation) of
        # the graph. Specifying all root variables in this list and leaving it empty are equivalent. This attribute
        # directly affects _initialize_graph_status_for_traversing function and indirectly affects _infer_all_shapes and
        # _prune functions.
        self.root_names = list()

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        """
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be produced
        :return: a string similar to the seed
        """
        if seed == "":
            raise ValueError("Name seed must be an non-empty string")

        # Make the seed meet C-style naming convention
        seed = re.sub(
            "[^0-9a-zA-Z]", "_", seed
        )  # Only alphabets and numbers are allowed
        if re.match("^[0-9]", seed):  # The first symbol cannot be a number
            seed = "_" + seed

        # If seed has never been seen, we return it as it is. Otherwise, we will append an number to make it unique.
        if seed not in existing_names:
            existing_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in existing_names:
                i += 1
            new_name = seed + str(i)
            existing_names.add(new_name)
            return new_name

    def get_unique_scope_name(self, seed):
        return Topology._generate_unique_name(seed, self.scope_names)

    def declare_scope(self, seed, parent_scopes=None):
        scope = Scope(
            self.get_unique_scope_name(seed),
            parent_scopes,
            self.variable_name_set,
            self.operator_name_set,
            self.target_opset,
        )
        self.scopes.append(scope)
        return scope

    def unordered_operator_iterator(self):
        for scope in self.scopes:
            for operator in scope.operators.values():
                yield operator

    def unordered_variable_iterator(self):
        for scope in self.scopes:
            for variable in scope.variables.values():
                yield variable

    def find_root_and_sink_variables(self):
        """
        Find root variables of the whole graph
        """
        # First we assume all variables are roots
        is_root = {
            name: True for scope in self.scopes for name in scope.variables.keys()
        }
        # Then, we remove those variables which are outputs of some operators
        for operator in self.unordered_operator_iterator():
            for variable in operator.outputs:
                is_root[variable.onnx_name] = False
        is_sink = {
            name: True for scope in self.scopes for name in scope.variables.keys()
        }
        for operator in self.unordered_operator_iterator():
            for variable in operator.inputs:
                is_sink[variable.onnx_name] = False
        return [
            variable
            for scope in self.scopes
            for name, variable in scope.variables.items()
            if is_root[name] or is_sink[name]
        ]

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in Topology object.
        Operators may be produced in a topological order. If you want to
        simply go though all operators without considering their
        topological structure, please use another function,
        unordered_operator_iterator.
        """
        self._initialize_graph_status_for_traversing()
        priorities = {"tensorToProbabilityMap": 2, "tensorToLabel": 1}
        while not all(
            operator.is_evaluated
            for scope in self.scopes
            for operator in scope.operators.values()
        ):
            is_evaluation_happened = False
            for operator in sorted(
                self.unordered_operator_iterator(),
                key=lambda op: priorities[op.type] if op.type in priorities else 0,
            ):
                if (
                    all(variable.is_fed for variable in operator.inputs)
                    and not operator.is_evaluated
                ):
                    # Check if over-writing problem occurs (i.e., multiple
                    # operators produce results on one variable).
                    for variable in operator.outputs:
                        # Throw an error if this variable has been treated as
                        # an output somewhere
                        if variable.is_fed:
                            raise RuntimeError(
                                "A variable is already assigned ({}) "
                                "for operator '{}' (name='{}'). This "
                                "may still happen if a converter is a "
                                "combination of sub-operators and one of "
                                "of them is producing this output. "
                                "In that case, an identity node must be "
                                "added.".format(
                                    variable, operator.type, operator.onnx_name
                                )
                            )
                        # Mark this variable as filled
                        variable.is_fed = True
                    # Make this operator as handled
                    operator.is_evaluated = True
                    is_evaluation_happened = True

                    # Send out an operator
                    yield operator

                    # This step may create new nodes if the
                    # the converter is called while looping on
                    # the nodes. The outputs of an operator
                    # are not necessary the inputs of the next
                    # one and but can processed by other ONNX nodes
                    # inserted in the container. As a result, some
                    # variables never have is_fed set to True which
                    # is updated now unless they are an operator
                    # output.
                    known_outputs = {}
                    for op in self.unordered_operator_iterator():
                        for out in op.outputs:
                            if hasattr(out, "onnx_name"):
                                known_outputs[out.onnx_name] = out
                            else:
                                known_outputs[out] = out
                    for variable in self.unordered_variable_iterator():
                        if variable.is_fed:
                            continue
                        if variable.onnx_name in known_outputs:
                            continue
                        update = (
                            False
                            if self.root_names
                            and variable.onnx_name not in self.root_names
                            else True
                        )
                        if update:
                            variable.is_fed = True
                            is_evaluation_happened = True

            # After scanning through the whole computational graph, at
            # least one operator should be evaluated. If not, we need
            # to terminate this procedure to avoid dead lock.
            if not is_evaluation_happened:
                break

    def rename_variable(self, old_name, new_name):
        """
        Replace the old ONNX variable name with a new ONNX variable name. There are several fields we need to edit.
            a. Topology
                1. scopes (the scope where the specified ONNX variable was declared)
                2. variable_name_set
            b. Scope
                1. onnx_variable_names (a mirror of Topology's variable_name_set)
                2. variable_name_mapping
                3. variables

        :param old_name: a string
        :param new_name: a string
        """
        # Search for the first variable that is named as old_name.
        scope, onnx_name, variable = next(
            (scope, onnx_name, variable)
            for scope in self.scopes
            for onnx_name, variable in scope.variables.items()
            if onnx_name == old_name
        )

        # Rename the variable we just found
        variable.onnx_name = new_name

        # Because the ONNX name of the targeted variable got changed, the (onnx_name, variable) pair in the associated
        # scope's variable dictionary should be changed as well. We therefore create a new pair to replace the old pair.
        scope.variables[new_name] = variable
        del scope.variables[old_name]

        # One original CoreML name may have several ONNX names recorded. To fix the record affected by renaming, we need
        # to replace old_name with new_name in the record of the associated CoreML name (variable.raw_name). Note that
        # derived_names contains all ONNX variable names derived from variable.raw_name.
        derived_names = scope.variable_name_mapping[variable.raw_name]
        for i in range(len(derived_names)):
            # Find old_name in derived_names
            if old_name != derived_names[i]:
                continue
            # Replace the recorded ONNX name with the new name
            derived_names[i] = new_name
            # Because ONNX names are unique so name replacement only happens once, we terminate the loop right after one
            # name replacement.
            break

        # Finally, new_name takes the place of old_name in the set of all existing variable names
        scope.onnx_variable_names.remove(old_name)
        scope.onnx_variable_names.add(new_name)

    def _check_structure(self):
        """
        This function applies some rules to check if the parsed model is proper. Currently, it only checks if isolated
        variable and isolated operator exists.
        """
        # Collect all variable names and operator names
        unused_variables = set()
        unused_operators = set()
        for variable in self.unordered_variable_iterator():
            unused_variables.add(variable.full_name)
        for operator in self.unordered_operator_iterator():
            unused_operators.add(operator.full_name)

        for operator in self.unordered_operator_iterator():
            for variable in operator.inputs:
                # A variable is used by an operator, so we remove the variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an input, so we remove the operator from the unused-operator list.
                unused_operators.discard(operator.full_name)
            for variable in operator.outputs:
                # A variable is used by an operator, so we remove the variable from the unused-variable list.
                unused_variables.discard(variable.full_name)
                # A operator has an output, so we remove the operator from the unused-operator list.
                unused_operators.discard(operator.full_name)

        if len(unused_variables) > 0:
            raise RuntimeError("Isolated variables exist: %s" % unused_variables)

        if len(unused_operators) > 0:
            raise RuntimeError("Isolated operators exist: %s" % unused_operators)

    def _initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators for traversing the underline graph
        """
        # In the beginning, we set is_root and is_leaf true. For is_fed, we have two different behaviors depending on
        # whether root_names is empty.
        for variable in self.unordered_variable_iterator():
            # If root_names is set, we only set those variable to be fed. Otherwise, all roots would be fed.
            if self.root_names:
                if variable.onnx_name in self.root_names:
                    variable.is_fed = True
                else:
                    variable.is_fed = False
            else:
                variable.is_fed = True
            variable.is_root = True
            variable.is_leaf = True

        # Then, we flip some flags by applying some simple rules so that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True
        for operator in self.unordered_operator_iterator():
            operator.is_evaluated = (
                False  # All operators are not processed in the beginning
            )
            for variable in operator.outputs:
                # Output cannot be fed before graph traversing
                variable.is_fed = False
                # If the variable is an output of one operator, it must not be a root
                variable.is_root = False
            for variable in operator.inputs:
                # If the variable is an input of one operator, it must not be a leaf
                variable.is_leaf = False

    def _infer_all_types(self):
        """
        Infer all variables' shapes in the computational graph.
        """
        self._initialize_graph_status_for_traversing()

        # Deliver user-specified types to root variables
        for raw_name, initial_type in self.initial_types:
            # Check all variables declared using raw_name in the whole graph
            for scope in self.scopes:
                # Skip scopes without having the considered variable name
                if raw_name not in scope.variable_name_mapping:
                    continue
                # Assign initial_type to all variables declared using raw_name
                for onnx_name in scope.variable_name_mapping[raw_name]:
                    variable = scope.variables[onnx_name]
                    if variable.is_root:
                        # Assign type to the root; existing type produced by parser may be overwritten
                        variable.type = initial_type

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            if operator.type in self.custom_shape_calculators:
                self.custom_shape_calculators[operator.type](operator)
            elif operator.type in self.custom_conversion_functions:
                pass  # in Keras converter, the shape calculator can be optional.
            else:
                operator.infer_types()

    def _resolve_duplicates(self):
        """
        Merge variables connected by identity operator to reduce the number of redundant variables
        """
        self._initialize_graph_status_for_traversing()

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            if operator.type != "identity":
                continue

            if any(variable.is_root for variable in operator.inputs) and any(
                variable.is_leaf for variable in operator.outputs
            ):
                continue

            # Replace the output variable with the input variable everywhere
            original = operator.inputs[0]
            duplicate = operator.outputs[0]
            for another_scope in self.scopes:
                for another_operator in another_scope.operators.values():
                    for i in range(len(another_operator.inputs)):
                        if another_operator.inputs[i].onnx_name != duplicate.onnx_name:
                            continue
                        another_operator.inputs[i] = original

            # When original variable's documentation string or denotation is empty but duplicate's is not, we
            # copy that field to the original variable to avoid information loss.
            if not original.type.doc_string and duplicate.type.doc_string:
                original.type.doc_string = duplicate.type.doc_string

            if isinstance(original.type, TensorType) and isinstance(
                duplicate.type, TensorType
            ):
                if not original.type.denotation and duplicate.type.denotation:
                    original.type.denotation = duplicate.type.denotation
                if not original.type.channel_denotations:
                    original.type.channel_denotations = (
                        duplicate.type.channel_denotations
                    )
                elif duplicate.type.channel_denotations:
                    # Merge the channel denotations if available in both the original and the duplicate
                    for i in range(len(original.type.channel_denotations)):
                        if original.type.channel_denotations[i]:
                            continue
                        original.type.channel_denotations[i] = (
                            duplicate.type.channel_denotations[i]
                        )
                # Sometime, shapes of duplicates are different.
                # We try to replace the original variable's unknown dimensions...
                # ...as many as possible because we will get rid of the duplicate.
                if len(original.type.shape) == len(duplicate.type.shape):
                    for i in range(len(original.type.shape)):
                        if original.type.shape[i] != "None":
                            continue
                        original.type.shape[i] = duplicate.type.shape[i]

            # Because we're iterating through the topology, we cannot delete any operator or variable. Otherwise,
            # the traversing function may be broken. We will delete those abandoned ones later.
            duplicate.is_abandoned = True
            operator.is_abandoned = True

        for scope in self.scopes:
            # Find out who is going to be abandoned
            abandoned_operator_names = set(
                onnx_name
                for onnx_name, operator in scope.operators.items()
                if operator.is_abandoned
            )
            abandoned_variable_names = set(
                onnx_name
                for onnx_name, variable in scope.variables.items()
                if variable.is_abandoned
            )

            # Remove abandoned operators
            for name in abandoned_operator_names:
                scope.delete_local_operator(name)

            # Remove abandoned variables
            for name in abandoned_variable_names:
                scope.delete_local_variable(name)

    def _fix_shapes(self):
        """
        This function applies some rules to adjust graph inputs (i.e., roots) before doing shape inference
        """

        # Identify roots of a graph
        self._initialize_graph_status_for_traversing()

        # Scan through all operators and adjust their variables' shapes if needed
        for operator in self.unordered_operator_iterator():
            # Rule 1 (CoreML):
            # Some operator in CoreML only accepts 4-D tensors but their protobuf models might specify a 2-D one.
            # We fix this problem here.
            if operator.type in [
                "bias",
                "concat",
                "convolution",
                "crop",
                "flatten",
                "scalerPreprocessor",
                "lrn",
                "meanImagePreprocessor",
                "padding",
                "permute",
                "pooling",
                "reduce",
                "reorganizeData",
                "reshape",
                "scale",
                "slice",
                "upsample",
            ]:
                # We only adjust inputs because outputs will be automatically fixed at our shape inference stage
                for variable in operator.inputs:
                    if variable.is_root:
                        # Convert [N, C] to [N, C, 1, 1] while [N, C, H, W] is unchanged
                        variable.type.shape += [1] * (4 - len(variable.type.shape))

    def _prune(self):
        # Conduct a dummy evaluation of this topology. It may set all reachable operators evaluated and all reachable
        # variables fed.
        for operator in self.topological_operator_iterator():
            pass

        for scope in self.scopes:
            # Remove unused operators
            abandoned_operator_names = []
            for operator in scope.operators.values():
                if not operator.is_evaluated:
                    abandoned_operator_names.append(operator.onnx_name)
            for onnx_name in abandoned_operator_names:
                scope.delete_local_operator(onnx_name)

            # Remove unused variables
            abandoned_variable_names = []
            for variable in scope.variables.values():
                if not variable.is_fed:
                    abandoned_variable_names.append(variable.onnx_name)
            for onnx_name in abandoned_variable_names:
                scope.delete_local_variable(onnx_name)

    def compile(self):
        """
        This function aims at giving every operator enough information so that all operator conversions can happen
        independently. We also want to check, fix, and simplify the network structure here.
        """
        self._prune()
        self._resolve_duplicates()
        self._fix_shapes()
        self._infer_all_types()
        self._check_structure()
