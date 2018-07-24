# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import re
from distutils.version import StrictVersion
from ...proto import onnx
from ...proto import helper
from .data_types import *
from ._container import ModelComponentContainer
from . import _registration
from . import utils
from .interface import OperatorBase

class Variable:

    def __init__(self, raw_name, onnx_name, scope, type=None):
        '''
        :param raw_name: A string indicating the variable's name in the original model. Usually, it's the seed string
        used to created its ONNX name (i.e., the field onnx_name below).
        :param onnx_name: A string indicating the variable's name in the converted model
        :param scope: A string. It's the name of the scope where this variable is declared
        :param type: A type object defined in onnxmltools.convert.common.data_types.py; e.g., FloatTensorType
        '''
        self.raw_name = raw_name  #
        self.onnx_name = onnx_name  #
        self.scope = scope
        self.type = type
        # The following fields are bool variables used in parsing and compiling stages
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


class Operator(OperatorBase):

    def __init__(self, onnx_name, scope, type, raw_operator, targeted_onnx_version):
        '''
        :param onnx_name: A unique ID, which is a string
        :param scope: The name of the scope where this operator is declared. It's a string.
        :param type: A object which uniquely characterizes the type of this operator. For example, it can be a string,
        pooling, if this operator is associated with a CoreML pooling layer.
        :param raw_operator: The original operator which defines this operator; for example, a scikit-learn Imputer and
        a CoreML Normalizer.
        :param targeted_onnx_version: A StrictVersion object indicating the ONNX version used
        '''
        self.onnx_name = onnx_name  # operator name in the converted model
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = []
        self.outputs = []
        self.is_evaluated = None
        self.is_abandoned = False
        self.targeted_onnx_version = targeted_onnx_version

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
        Return all output variables' names
        '''
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        '''
        Return the original operator/layer
        '''
        return self.raw_operator

    def infer_types(self):
        # Invoke a core inference function
        _registration.get_shape_calculator(self.type)(self)


class Scope:

    def __init__(self, name, parent_scopes=None, variable_name_set=None, operator_name_set=None,
                 targeted_onnx_version=None):
        '''
        :param name:  A string, the unique ID of this scope in a Topology object
        :param parent_scopes: A list of Scope objects. The last element should be the direct parent scope (i.e., where
        this scope is declared).
        :param variable_name_set: A set of strings serving as the name pool of variables
        :param operator_name_set: A set of strings serving as the name pool of operators
        :param targeted_onnx_version: A StrictVersion object indicating the ONNX version used
        '''
        self.name = name
        self.parent_scopes = parent_scopes if parent_scopes else list()
        self.onnx_variable_names = variable_name_set if variable_name_set is not None else set()
        self.onnx_operator_names = operator_name_set if operator_name_set is not None else set()
        self.targeted_onnx_version = targeted_onnx_version

        # An one-to-many map from raw variable name to ONNX variable names. It looks like
        #   (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ..., onnx_nameN])
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope. (key, value) = (onnx_name, variable)
        self.variables = {}

        # A map of local operators defined in this scope. (key, value) = (onnx_name, operator)
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
        named in self.variable_name_mapping[raw_name] will be returned. Otherwise, a new variable will be created and
        then returned.
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
        onnx_name = self.get_unique_operator_name(str(type))
        operator = Operator(onnx_name, self.name, type, raw_model, self.targeted_onnx_version)
        self.operators[onnx_name] = operator
        return operator

    def delete_local_operator(self, onnx_name):
        '''
        Remove the operator whose onnx_name is the input onnx_name
        '''
        if onnx_name not in self.onnx_operator_names or onnx_name not in self.operators:
            raise RuntimeError('The operator to be removed not found')
        self.onnx_operator_names.discard(onnx_name)
        del self.operators[onnx_name]

    def delete_local_variable(self, onnx_name):
        '''
        Remove the variable whose onnx_name is the input onnx_name
        '''
        if onnx_name not in self.onnx_variable_names or onnx_name not in self.variables:
            raise RuntimeError('The variable to be removed not found')
        self.onnx_variable_names.discard(onnx_name)
        raw_name = self.variables[onnx_name].raw_name
        self.variable_name_mapping[raw_name].remove(onnx_name)
        del self.variables[onnx_name]


class Topology:

    def __init__(self, model, default_batch_size=1, initial_types=None,
                 reserved_variable_names=None, reserved_operator_names=None, targeted_onnx=None,
                 custom_conversion_functions=None, custom_shape_calculators=None):
        '''
        Initialize a Topology object, which is an intermediate representation of a computational graph.

        :param model: RawModelContainer object or one of its derived classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and array types from CoreML. It's usually 1 or 'None'.
        :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
        name and a type defined in data_types.py.
        :param reserved_variable_names: A set of strings which are not allowed to be used as a variable name
        :param reserved_operator_names: A set of strings which are not allowed to be used as a operator name
        :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
        '''
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.variable_name_set = reserved_variable_names if reserved_variable_names is not None else set()
        self.operator_name_set = reserved_operator_names if reserved_operator_names is not None else set()
        self.initial_types = initial_types if initial_types else list()
        self.default_batch_size = default_batch_size
        self.targeted_onnx_version = StrictVersion(targeted_onnx)
        self.custom_conversion_functions = custom_conversion_functions if custom_conversion_functions else {}
        self.custom_shape_calculators = custom_shape_calculators if custom_shape_calculators else {}

        # This attribute is used in optimizing the graph structure. If root_names is not empty, only the variables
        # specified will be treated as the roots (i.e., set is_fed to True in the beginning of a graph evaluation) of
        # the graph. Specifying all root variables in this list and leaving it empty are equivalent. This attribute
        # directly affects _initialize_graph_status_for_traversing function and indirectly affects _infer_all_shapes and
        # _prune functions.
        self.root_names = list()

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        '''
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be produced
        :return: a string similar to the seed
        '''
        if seed == '':
            raise ValueError('Name seed must be an non-empty string')

        # Make the seed meet C-style naming convention
        seed = re.sub('[^0-9a-zA-Z]', '_', seed)  # Only alphabets and numbers are allowed
        if re.match('^[0-9]', seed):  # The first symbol cannot be a number
            seed = '_' + seed

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

    def declare_scope(self, seed, parent_scopes=list()):
        scope = Scope(self.get_unique_scope_name(seed), parent_scopes, self.variable_name_set,
                      self.operator_name_set, self.targeted_onnx_version)
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
        priorities = {'tensorToProbabilityMap': 2, 'tensorToLabel': 1}
        while not all(operator.is_evaluated for scope in self.scopes for operator in scope.operators.values()):
            is_evaluation_happened = False
            for operator in sorted(self.unordered_operator_iterator(),
                                   key=lambda op: priorities[op.type] if op.type in priorities else 0):
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
        '''
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
        '''
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
        '''
        Merge variables connected by identity operator to reduce the number of redundant variables
        '''
        self._initialize_graph_status_for_traversing()

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():
            if operator.type != 'identity':
                continue

            if any(variable.is_root for variable in operator.inputs) and \
                    any(variable.is_leaf for variable in operator.outputs):
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

            # When original variable's document string is empty but duplicate's document string is not, we
            # copy that non-empty string to the original variable to avoid information loss.
            if not original.type.doc_string and duplicate.type.doc_string:
                original.type.doc_string = duplicate.type.doc_string

            # Sometime, shapes of duplicates are different. We try to replace the original variable's unknown dimensions
            # as many as possible because we will get rid of the duplicate.
            if isinstance(original.type, TensorType) and isinstance(duplicate.type, TensorType) and \
                    len(original.type.shape) == len(duplicate.type.shape):
                for i in range(len(original.type.shape)):
                    if original.type.shape[i] != 'None':
                        continue
                    original.type.shape[i] = duplicate.type.shape[i]

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
                scope.delete_local_operator(name)

            # Remove abandoned variables
            for name in abandoned_variable_names:
                scope.delete_local_variable(name)

    def _fix_shapes(self):
        '''
        This function applies some rules to adjust graph inputs (i.e., roots) before doing shape inference
        '''

        # Identify roots of a graph
        self._initialize_graph_status_for_traversing()

        # Scan through all operators and adjust their variables' shapes if needed
        for operator in self.unordered_operator_iterator():
            # Rule 1 (CoreML):
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
        '''
        This function aims at giving every operator enough information so that all operator conversions can happen
        independently. We also want to check, fix, and simplify the network structure here.
        '''
        self._prune()
        self._resolve_duplicates()
        self._fix_shapes()
        self._infer_all_types()
        self._check_structure()


def convert_topology(topology, model_name, doc_string, targeted_onnx):
    '''
    This function is used to convert our Topology object defined in _parser.py into a ONNX model (type: ModelProto).
    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the returned model. The string "model_name" would be
    assigned to "model.graph.name."
    :param doc_string: A string attached to the produced model
    :param targeted_onnx: A string, which specifies the targeted ONNX version of the produced model. Possible values
    include '1.1.2', '1.2', and so on.
    :return: a ONNX ModelProto
    '''
    if targeted_onnx != onnx.__version__:
        raise RuntimeError(
            'ONNX version conflict found. The installed version is %s while the targeted version is %s' % (
                onnx.__version__, targeted_onnx))

    topology._initialize_graph_status_for_traversing()

    container = ModelComponentContainer(targeted_onnx)

    # Put roots and leaves as ONNX's model into buffers. They will be added into ModelComponentContainer later.
    tensor_inputs = {}
    other_inputs = {}
    tensor_outputs = {}
    other_outputs = {}
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                if isinstance(variable.type, (TensorType, Int64Type, FloatType, StringType)):
                    tensor_inputs[variable.raw_name] = variable
                else:
                    other_inputs[variable.raw_name] = variable
            if variable.is_leaf:
                if isinstance(variable.type, (TensorType, Int64Type, FloatType, StringType)):
                    tensor_outputs[variable.raw_name] = variable
                else:
                    other_outputs[variable.raw_name] = variable

    # Add roots the graph according to their order in the original model
    for name in topology.raw_model.input_names:
        if name in tensor_inputs:
            container.add_input(tensor_inputs[name])
    for name in topology.raw_model.input_names:
        if name in other_inputs:
            container.add_input(other_inputs[name])

    # Add leaves the graph according to their order in the original model
    for name in topology.raw_model.output_names:
        if name in tensor_outputs:
            container.add_output(tensor_outputs[name])
    for name in topology.raw_model.output_names:
        if name in other_outputs:
            container.add_output(other_outputs[name])

    # Traverse the graph from roots to leaves
    for operator in topology.topological_operator_iterator():
        scope = next(scope for scope in topology.scopes if scope.name == operator.scope)
        if operator.type in topology.custom_conversion_functions:
            topology.custom_conversion_functions[operator.type](scope, operator, container)
        else:
            # Convert the selected operator into some ONNX objects and save them into the container
            _registration.get_converter(operator.type)(scope, operator, container)

    # When calling ModelComponentContainer's add_initializer(...), nothing is added into the input list. However, in
    # ONNX initializers should also be model's (GraphProto) inputs. Thus, we create ValueInfoProto objects from
    # initializers (type: TensorProto) directly and then add them into model's input list.
    extra_inputs = []  # ValueInfoProto list of the initializers
    for tensor in container.initializers:
        # Sometimes (especially when creating optional input values such as RNN's initial hidden state), an initializer
        # is also one of the original model's input, so it has been added into the container's input list. If this is
        # the case, we need to skip one iteration to avoid duplicated inputs.
        if tensor.name in [value_info.name for value_info in container.inputs]:
            continue

        # Initializers are always tensors so we can just call make_tensor_value_info(...)
        value_info = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
        extra_inputs.append(value_info)

    # Create a graph from its main components
    graph = helper.make_graph(container.nodes, model_name, container.inputs + extra_inputs,
                              container.outputs, container.initializers)

    # Add extra information related to the graph
    graph.value_info.extend(container.value_info)

    # Create model
    onnx_model = helper.make_model(graph)

    # Merge operator sets for the same domain, the largest version number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in container.node_domain_version_pair_sets:
        if op_domain not in purified_operator_set:
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(purified_operator_set[op_domain], op_version)

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by helper.make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
        i += 1

    # Add extra information
    onnx_model.ir_version = onnx_proto.IR_VERSION
    onnx_model.producer_name = utils.get_producer()
    onnx_model.producer_version = utils.get_producer_version()
    onnx_model.domain = utils.get_domain()
    onnx_model.model_version = utils.get_model_version()
    onnx_model.doc_string = doc_string

    return onnx_model
