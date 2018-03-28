# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...proto import helper
from . import registration
from ._data_types import *
from ._container import ModelComponentContainer


class RawModelContainer(object):

    def __init__(self, raw_model):
        self._raw_model = raw_model

    @property
    def raw_model(self):
        return self._raw_model

    @property
    def input_names(self):
        raise NotImplementedError()

    @property
    def output_names(self):
        raise NotImplementedError()


class CoremlModelContainer(RawModelContainer):

    def __init__(self, coreml_model):
        super(CoremlModelContainer, self).__init__(coreml_model)

    @property
    def input_names(self):
        return [str(var.name) for var in self.raw_model.description.input]

    @property
    def output_names(self):
        return [str(var.name) for var in self.raw_model.description.output]


class SklearnModelContainer(RawModelContainer):

    def __init__(self, sklearn_model):
        super(SklearnModelContainer, self).__init__(sklearn_model)
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        self._inputs.append(variable)

    def add_output(self, variable):
        self._outputs.append(variable)


class Variable:

    def __init__(self, raw_name, onnx_name, scope, type=None):
        self.raw_name = raw_name  # variable name in the original model
        self.onnx_name = onnx_name  # variable name in the converted model
        self.scope = scope
        self.type = type
        self.is_fed = None
        self.is_root = None
        self.is_leaf = None
        self.is_abandoned = False

    @property
    def full_name(self):
        '''
        Return a globally unique variable ID
        '''
        return self.onnx_name


class Operator:

    def __init__(self, onnx_name, scope, type, raw_operator):
        self.onnx_name = onnx_name  # operator name in the converted model
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = []
        self.outputs = []
        self.is_evaluated = None
        self.is_abandoned = False

    @property
    def full_name(self):
        '''
        Return a globally unique operator ID
        '''
        return self.onnx_name

    @property
    def input_full_names(self):
        '''
        Return all input variables' names
        '''
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        '''
        Return all outpu variables' names
        '''
        return [variable.full_name for variable in self.outputs]

    def infer_types(self):
        # Invoke a core inference function
        registration.get_shape_calculator(self.type)(self)


class Scope:

    def __init__(self, name, parent_scopes=None, variable_name_set=None, operator_name_set=None):
        # name: scope's ID. It's unique in a topology.
        # parent_scopes: all parents of this scope. It encodes the tree structure of the computational graph.
        # variable_name_set: set used to stored variable names declared in this scope.
        # operator_name_set: set used to stored operator names declared in this scope.
        # onnx_variable_names: variable IDs live in this scope.
        # onnx_operator_names: operator IDs live in this scope.
        # variables: a map of local variables defined in this scope. (key, value) = (onnx_name, variable)
        # operators: a map of local operators defined in this scope. (key, value) = (onnx_name, operator)
        # variable_name_mapping: an one-to-many map from raw variable name to ONNX variable names.
        #                        (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ...])
        # operator_name_mapping: an one-to-many map from raw operator name to ONNX operator names.
        #                        (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ...])
        self.name = name
        self.parent_scopes = parent_scopes if parent_scopes else list()
        self.onnx_variable_names = variable_name_set if variable_name_set is not None else set()
        self.onnx_operator_names = operator_name_set if operator_name_set is not None else set()
        self.variable_name_mapping = {}
        self.operator_name_mapping = {}
        self.variables = {}
        self.operators = {}

    def get_onnx_variable_name(self, seed):
        '''
        Retrieve the variable ID of the given seed or create one if it is the first time of seeing this seed
        '''
        if seed in self.variable_name_mapping:
            return self.variable_name_mapping[seed][-1]
        else:
            return self.get_unique_variable_name(seed)

    def get_unique_variable_name(self, seed):
        '''
        Create a unique variable ID based on the given seed
        '''
        return Topology._generate_unique_name(seed, self.onnx_variable_names)

    def get_unique_operator_name(self, seed):
        '''
        Create a unique operator ID based on the given seed
        '''
        return Topology._generate_unique_name(seed, self.onnx_operator_names)

    def find_sink_variables(self):
        '''
        Find sink variables in this scope
        '''
        # First we assume all variables are sinks
        is_sink = {name: True for name in self.variables.keys()}
        # Then, we remove those variables which are inputs of some operators
        for operator in self.operators.values():
            for variable in operator.inputs:
                is_sink[variable.onnx_name] = False
        return [variable for name, variable in self.variables.items() if is_sink[name]]

    def declare_local_variable(self, raw_name, type=None, prepend=False):
        '''
        This function may create a new variable in this scope. If raw_name has been used to create other variables,
        the new variable will hide all other variables created using raw_name.
        '''
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
        '''
        This function will first check if raw_name has been used to create some variables. If yes, the latest one
        will be returned. Otherwise, a new variable will be created and then returned.
        '''
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
        '''
        This function is used to declare new local operator.
        '''
        onnx_name = self.get_unique_operator_name(type)
        operator = Operator(onnx_name, self.name, type, raw_model)
        self.operators[onnx_name] = operator
        return operator


class Topology:

    def __init__(self, model, default_batch_size=1, initial_types=None,
                 reserved_variable_names=None, reserved_operator_names=None):
        '''
        Initialize a Topology object, which is an intermediate representation of a computational graph.

        :param model: the model used to create the topology
        :param default_batch_size: batch_size prepend to scalar and array types from CoreML
        :param initial_types: a dictionary providing some types for some CoreML root variables
        :param reserved_variable_names: a set of strings which are not allowed to be used as a variable name
        :param reserved_operator_names: a set of strings which are not allowed to be used as a operator name
        '''
        self.scopes = []
        self.raw_model = model
        self.scope_names = set('__none__')
        self.variable_name_set = reserved_variable_names if reserved_variable_names is not None else set()
        self.operator_name_set = reserved_operator_names if reserved_operator_names is not None else set()
        self.initial_types = initial_types if initial_types else dict()
        self.default_batch_size = default_batch_size

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        '''
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be produced
        :return: a string similar to the seed
        '''
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

    def declare_scope(self, seed, parent_scopes=list()):
        scope = Scope(self.get_unique_scope_name(seed), parent_scopes, self.variable_name_set, self.operator_name_set)
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
        '''
        Find root variables of the whole graph
        '''
        # First we assume all variables are roots
        is_root = {name: True for scope in self.scopes for name in scope.variables.keys()}
        # Then, we remove those variables which are outputs of some operators
        for operator in self.unordered_operator_iterator():
            for variable in operator.outputs:
                is_root[variable.onnx_name] = False
        is_sink = {name: True for scope in self.scopes for name in scope.variables.keys()}
        for operator in self.unordered_operator_iterator():
            for variable in operator.inputs:
                is_sink[variable.onnx_name] = False
        return [variable for scope in self.scopes for name, variable in scope.variables.items()
                if is_root[name] or is_sink[name]]

    def topological_operator_iterator(self):
        '''
        This is an iterator of all operators in Topology object. Operators may be produced in a topological order.
        If you want to simply go though all operators without considering their topological structure, please use
        another function, unordered_operator_iterator.
        '''
        self._initialize_graph_status_for_traversing()
        while not all(operator.is_evaluated for scope in self.scopes for operator in scope.operators.values()):
            is_evaluation_happened = False
            for scope in self.scopes:
                for operator in scope.operators.values():
                    if all(variable.is_fed for variable in operator.inputs) and not operator.is_evaluated:
                        # Check if over-writing problem occurs (i.e., multiple operators produce results on one variable).
                        for variable in operator.outputs:
                            # Throw an error if this variable has been treated as an output somewhere
                            if variable.is_fed:
                                raise RuntimeError('One variable can only be assigned once')
                            # Mark this variable as filled
                            variable.is_fed = True
                        # Make this operator as handled
                        operator.is_evaluated = True
                        is_evaluation_happened = True
                        # Send out an operator
                        yield operator
            # After scanning through the whole computational graph, at least one operator should be evaluated. If not,
            # we need to terminate this procedure to avoid dead lock.
            if not is_evaluation_happened:
                break

    def rename_variable(self, old_name, new_name):
        # Search for the first variable that is named as old_name.
        scope, onnx_name, variable = next((scope, onnx_name, variable) for scope in self.scopes
                                          for onnx_name, variable in scope.variables.items() if onnx_name == old_name)

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
            # Check if the ith ONNX name is just replaced by new_name
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
        '''
        This function applies some rules to check if the parsed model is proper. Currently, it only checks if isolated
        variable and isolated operator exists.
        '''
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
            raise RuntimeError('Isolated variables exist: %s' % unused_variables)

        if len(unused_operators) > 0:
            raise RuntimeError('Isolated operators exist: %s' % unused_operators)

    def _initialize_graph_status_for_traversing(self):
        '''
        Initialize the status of all variables and operators for traversing the underline graph
        '''
        # In the beginning, we set all flags true
        for variable in self.unordered_variable_iterator():
            variable.is_fed = True
            variable.is_root = True
            variable.is_leaf = True
        # Then, we flip some flags by applying some simple rules so that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True
        for operator in self.unordered_operator_iterator():
            operator.is_evaluated = False  # All operators are not processed in the beginning
            for variable in operator.outputs:
                # Output cannot be fed before graph traversing
                variable.is_fed = False
                # If the variable is an output of one operator, it must not be a root
                variable.is_root = False
            for variable in operator.inputs:
                # If the variable is an input of one operator, it must not be a leaf
                variable.is_leaf = False

    def _infer_all_types(self):
        '''
        Infer all variables' shapes in the computational graph.
        '''
        self._initialize_graph_status_for_traversing()

        # Deliver user-specified types to root variables
        for raw_name, initial_type in self.initial_types.items():
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
            operator.infer_types()

    def _resolve_duplicates(self):
        '''
        Merge variables connected by identity operator to reduce the number of redundant variables
        '''
        self._initialize_graph_status_for_traversing()

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            if operator.type != 'identity':
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
            # Because we're iterating through the topology, we cannot delete any operator or variable. Otherwise,
            # the traversing function may be broken. We will delete those abandoned ones later.
            duplicate.is_abandoned = True
            operator.is_abandoned = True

        for scope in self.scopes:
            # Find out who is going to be abandoned
            abandoned_operator_names = set(onnx_name for onnx_name, operator in scope.operators.items()
                                           if operator.is_abandoned)
            abandoned_variable_names = set(onnx_name for onnx_name, variable in scope.variables.items()
                                           if variable.is_abandoned)
            # Remove abandoned operators
            for name in abandoned_operator_names:
                scope.onnx_operator_names.discard(name)  # this variable is a global structure shared by all scopes!
                if name in scope.operator_name_mapping:
                    del scope.operator_name_mapping[name]
                if name in scope.operators:
                    del scope.operators[name]
            # Remove abandoned variables
            for name in abandoned_variable_names:
                scope.onnx_variable_names.discard(name)  # this variable is a global structure shared by all scopes!
                if name in scope.variable_name_mapping:
                    del scope.variable_name_mapping[name]
                if name in scope.variables:
                    del scope.variables[name]

    def _fix_shapes(self):
        '''
        This function applies some rules to adjust graph inputs (i.e., roots) before doing shape inference
        '''

        # Identify roots of a graph
        self._initialize_graph_status_for_traversing()

        # Scan through all operators and adjust their variables' shapes if needed
        for operator in self.unordered_operator_iterator():
            # Rule 1:
            # Some operator in CoreML only accepts 4-D tensors but their protobuf models might specify a 2-D one.
            # We fix this problem here.
            if operator.type in ['bias', 'concat', 'convolution', 'crop', 'flatten', 'scalerPreprocessor',
                                 'lrn', 'meanImagePreprocessor', 'padding', 'permute', 'pooling', 'reduce',
                                 'reorganizeData', 'reshape', 'scale', 'slice', 'upsample']:
                # We only adjust inputs because outputs will be automatically fixed at our shape inference stage
                for variable in operator.inputs:
                    if variable.is_root:
                        # Convert [N, C] to [N, C, 1, 1] while [N, C, H, W] is unchanged
                        variable.type.shape += [1] * (4 - len(variable.type.shape))

            # Rule 2:
            # Some model in ONNX accepts integers while the corresponding one in CoreML only takes floats.
            # If it is the case, we change tensor type from float to integer.
            if operator.type == 'embedding':
                for variable in operator.inputs:
                    if variable.is_root:
                        variable.type = Int64TensorType(variable.type.shape, doc_string=variable.type.doc_string)
                    else:
                        raise RuntimeError('Embed operator in ONNX only accepts floats but we got integers')

    def compile(self):
        '''
        This function aims at giving every operator enough information so that all operator conversions can happen
        independently. We also want to check, fix, and simplify the network structure here.
        '''
        self._check_structure()
        self._resolve_duplicates()
        self._fix_shapes()
        self._infer_all_types()


def convert_topology(topology, model_name):
    '''
    This function is used to convert our Topology object defined in _parser.py into a ONNX model (type: ModelProto).
    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the output model. The string "model_name" would be assigned
    to "model.graph.name."
    :return: a ONNX ModelProto
    '''
    topology._initialize_graph_status_for_traversing()

    container = ModelComponentContainer()

    # Add roots and leaves as ONNX's model inputs and outputs
    model_inputs = []
    model_outputs = []
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                model_inputs.append(variable)
            if variable.is_leaf:
                model_outputs.append(variable)
    # Add roots and leaves of the graph according to their order in the original CoreML model
    for name in topology.raw_model.input_names:
        variable = next(variable for variable in model_inputs if variable.raw_name == name)
        container.add_input(variable)
    for name in topology.raw_model.output_names:
        variable = next(variable for variable in model_outputs if variable.raw_name == name)
        container.add_output(variable)

    # Traverse the graph from roots to leaves
    for operator in topology.topological_operator_iterator():
        # Convert the selected operator into some ONNX objects and save them into the container
        registration.get_converter(operator.type)(scope, operator, container)

    # Move ZipMap nodes to the end of the node list. In the future, here should be a sorting function which re-orders
    # the nodes according the model's outputs.
    for i, node in enumerate(container.nodes):
        if node.op_type != 'ZipMap':
            continue
        zipmap_node = container.nodes[i]
        for another_node_id in range(i + 1, len(container.nodes)):
            another_node = container.nodes[another_node_id]
            if zipmap_node.output[0] not in another_node.input:
                container.nodes[i], container.nodes[another_node_id] = \
                    container.nodes[another_node_id], container.nodes[i]

    # Create a graph from its main components
    graph = helper.make_graph(container.nodes, model_name, container.inputs, container.outputs, container.initializers)
    # Add extra information related to the graph
    graph.value_info.extend(container.value_info)
    onnx_model = helper.make_model(graph)
    for op_domain, op_version in container.node_domain_version_pair_sets:
        op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
    return onnx_model
