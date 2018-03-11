from ._data_types import *
from ._shape_calculators import type_calculator_table
from graphviz import Digraph


class Variable:

    def __init__(self, raw_name, onnx_name, scope, type=None):
        self.raw_name = raw_name
        self.onnx_name = onnx_name
        self.scope = scope
        self.type = type
        self.fed_order = None
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
        self.onnx_name = onnx_name
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
        if self.type in type_calculator_table:
            type_calculator_table[self.type](self)
        else:
            raise RuntimeError('Shape calculator of operator %s not found' % self.type)


class Scope:

    def __init__(self, name, parent_scopes=[], variable_name_set=set(), operator_name_set=set()):
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
        self.parent_scopes = parent_scopes
        self.onnx_variable_names = variable_name_set
        self.onnx_operator_names = operator_name_set
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
        if seed not in self.onnx_variable_names:
            self.onnx_variable_names.add(seed)
            self.variable_name_mapping[seed] = [seed]
            return seed
        else:
            i = 1
            new_name = seed + str(i)
            while seed + str(i) in self.onnx_variable_names:
                i += 1
            new_name = seed + str(i)
            self.onnx_variable_names.add(new_name)
            return new_name

    def get_unique_operator_name(self, seed):
        '''
        Create a unique operator ID based on the given seed
        '''
        if seed not in self.onnx_operator_names:
            self.onnx_operator_names.add(seed)
            self.operator_name_mapping[seed] = [seed]
            return seed
        else:
            i = 1
            while seed + str(i) in self.onnx_operator_names:
                i += 1
            new_name = seed + str(i)
            self.onnx_operator_names.add(new_name)
            return new_name

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

    def __init__(self, model, default_batch_size=1, initial_types={},
                 reserved_variable_names=set(), reserved_operator_names=set()):
        '''
        Initialize a Topology object, which is an intermediate representation of a computational graph

        :param model: the model used to create the topology
        :param default_batch_size: batch_size prepend to scalar and array types from CoreML
        :param initial_types: a dictionary providing some types for some CoreML root variables
        :param reserved_variable_names: a set of strings which are not allowed to be used as a variable name
        :param reserved_operator_names: a set of strings which are not allowed to be used as a operator name
        '''
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.variable_name_set = reserved_variable_names
        self.operator_name_set = reserved_operator_names
        self.initial_types = initial_types
        self.default_batch_size = default_batch_size

    def get_unique_scope_name(self, seed):
        if seed not in self.scope_names:
            self.scope_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in self.scope_names:
                i += 1
            name = seed + str(i)
            self.scope_names.add(name)
            return name

    def declare_scope(self, seed, parent_scopes=[]):
        scope = Scope(self.get_unique_scope_name(seed), parent_scopes, self.variable_name_set, self.operator_name_set)
        self.scopes.append(scope)
        return scope

    def find_root_and_sink_variables(self):
        '''
        Find root variables of the whole graph
        '''
        # First we assume all variables are roots
        is_root = {name: True for scope in self.scopes for name in scope.variables.keys()}
        # Then, we remove those variables which are outputs of some operators
        for scope in self.scopes:
            for operator in scope.operators.values():
                for variable in operator.outputs:
                    is_root[variable.onnx_name] = False
        is_sink = {name: True for scope in self.scopes for name in scope.variables.keys()}
        for scope in self.scopes:
            for operator in scope.operators.values():
                for variable in operator.inputs:
                    is_sink[variable.onnx_name] = False
        return [variable for scope in self.scopes for name, variable in scope.variables.items()
                if is_root[name] or is_sink[name]]

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
        This function applies some rules to check if the parsed model is proper
        '''
        # variable over-writing (one variable is a output of multiple operators)
        # Isolated variables
        # Isolated operators
        pass

    def _initialize_graph_status_for_traversing(self):
        '''
        Initialize the status of all variables and operators for traversing the underline graph
        '''
        # In the beginning, we set all flags true
        for scope in self.scopes:
            for variable in scope.variables.values():
                variable.is_fed = True
                variable.is_root = True
                variable.is_leaf = True
        # Then, we flip some flags by applying some simple rules so that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True
        for scope in self.scopes:
            for operator in scope.operators.values():
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

        # [TODO] Apply default rules to pre-process root types
        # 1. If the top-level model is a neural network and the initial shape dictionary is empty, change all roots to
        #    4-D tensors with unknown batch size.
        # 2. For hidden and cell states RNN/LSTM/GRU, their batch sizes are one.
        # 3. We need to carefully revise the change of leaves' types. For example, the color space of an output image
        #    cannot be affected by our code.
        # 4. The input of embedding is integer tensor

        # [TODO] Fix shapes of top-level inputs and outputs to produce identical shape like CoreML
        # We use 4-D tensor, [N, C, H, W], as the major communication inferface between tensor operations. In CoreML,
        # some redundant coordinates may be squeezed; for example,[1, C, 1, 1] is usually encoded by [1, C] or [C].
        # Steps:
        #   1. Find roots and leaves of the graph
        #   2. Extract roots' and leaves' types from CoreML modle file
        #   3. Fix them if necessary

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
        while not all(operator.is_evaluated for scope in self.scopes for operator in scope.operators.values()):
            for scope in self.scopes:
                for operator in scope.operators.values():
                    if all(variable.is_fed for variable in operator.inputs) and not operator.is_evaluated:
                        # Infer output types
                        operator.infer_types()

                        # Check if over-writing problem occurs (i.e., multiple operators produce results on one variable).
                        for variable in operator.outputs:
                            # Throw an error if this variable has been treated as an output somewhere
                            if variable.is_fed:
                                raise RuntimeError('One variable can only be assigned once')
                            # Mark this variable as filled
                            variable.is_fed = True

                        # Make this operator as handled
                        operator.is_evaluated = True

    def _resolve_duplicates(self):
        '''
        Merge variables connected by identity operator to reduce the number of redundant variables
        '''
        self._initialize_graph_status_for_traversing()

        # Traverse the graph from roots to leaves
        while not all(operator.is_evaluated for scope in self.scopes for operator in scope.operators.values()):
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

                        if operator.type != 'identity':
                            continue
                        original = operator.inputs[0]
                        duplicate = operator.outputs[0]
                        for another_scope in self.scopes:
                            for another_operator in another_scope.operators.values():
                                for i in range(len(another_operator.inputs)):
                                    if another_operator.inputs[i].onnx_name != duplicate.onnx_name:
                                        continue
                                    another_operator.inputs[i] = original
                        duplicate.is_abandoned = True
                        operator.is_abandoned = True

        for scope in self.scopes:
            # Find out who is going to be abandoned
            abandoned_operator_names = set(onnx_name for onnx_name, operator in scope.operators.items()
                                           if operator.is_abandoned)
            abandoned_variable_names = set(onnx_name for onnx_name, variable in scope.variables.items()
                                           if variable.is_abandoned)
            for name in abandoned_operator_names:
                scope.onnx_operator_names.discard(name) # this variable is a global structure shared by all scopes!
                if name in scope.operator_name_mapping:
                    del scope.operator_name_mapping[name]
                if name in scope.operators:
                    del scope.operators[name]
            for name in abandoned_variable_names:
                scope.onnx_variable_names.discard(name) # this variable is a global structure shared by all scopes!
                if name in scope.variable_name_mapping:
                    del scope.variable_name_mapping[name]
                if name in scope.variables:
                    del scope.variables[name]

    def compile(self):
        '''
        This function aims at giving every operator enough information so that all operator conversions can happen independently.
        We also want to simplify the network structure here.
        '''
        self._check_structure()
        self._resolve_duplicates()
        self._infer_all_types()


def visualize_topology(topology, filename=None, view=False):
    # [TODO] Replace graphvis package with other choice
    graph = Digraph()

    # declare nodes (variables and operators)
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                graph.attr('node', shape='oval', style='filled', fillcolor='blue')
            else:
                graph.attr('node', shape='oval', style='filled', fillcolor='white')
            graph.node(variable.full_name, label=variable.full_name + ', ' + str(variable.type))

        for operator in scope.operators.values():
            graph.attr('node', shape='box', style='filled', fillcolor='green')
            graph.node(operator.full_name, label=operator.full_name)

    # declare edges (connections between variables and operators)
    for scope in topology.scopes:
        for operator in scope.operators.values():
            for variable in operator.inputs:
                graph.edge(variable.full_name, operator.full_name)
            for variable in operator.outputs:
                graph.edge(operator.full_name, variable.full_name)

    if filename is not None:
        graph.render(filename, view=view)

    return graph


def _parse_model(topology, scope, model, inputs=[], outputs=[]):
    '''
    This is a delegate function of all top-level parsing functions. It does nothing but call a proper function
    to parse the given model.
    '''

    model_type = model.WhichOneof('Type')
    if model_type in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        _parse_pipeline_model(topology, scope, model, inputs, outputs)
    elif model_type in ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork']:
        _parse_neural_network_model(topology, scope, model, inputs, outputs)
    else:
        _parse_simple_model(topology, scope, model, inputs, outputs)


def _parse_simple_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a model containing only one operator (aka simple model).
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Create operator and then feed the model's inputs and outputs to the operator
        3. Connect local variables and their corresponding parent variables
    Note:
        1. Notice that a CoreML operator can contain no input and output, so we directly use model's inputs (outputs).
        2. Input and output names can be identical in CoreML, but they must be different for ONNX.
    '''

    # Create local scope for the considered model
    scope = topology.declare_scope('single', [parent_scope] + parent_scope.parent_scopes)

    # Create operator for the considered model
    this_operator = scope.declare_local_operator(model.WhichOneof('Type'), model)

    # [TODO] Add feature vectorizer into all classifiers/regressors
    # Allocate inputs for the operator and then connect them with inputs from outside
    for var in model.description.input:
        # We assume that no duplicated raw name exists. Note that we set prepend=True because model inputs should
        # not hide any intermediate variables.
        variable = scope.declare_local_variable(
            var.name,parse_coreml_feature_type(var.type, topology.default_batch_size), prepend=True)
        this_operator.inputs.append(variable)
    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the first one to parent input.

    # [TODO] Add feature vectorizer if there are multiple feature tensors
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Allocate outputs for the operator and then connect them with outputs from outside
    for var in model.description.output:
        # We assume that no duplicated output raw name exists.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature_type(var.type, topology.default_batch_size))
        this_operator.outputs.append(variable)
    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the last one to parent output.
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_pipeline_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a pipeline including multiple sub-models.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the sub-models and create their inputs and outputs variables
        3. Connect model's (not sub-model's) inputs and outputs with proper variables created when parsing sub-models
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML sub-model can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
    '''

    # Create local scope
    scope = topology.declare_scope('pipeline', [parent_scope] + parent_scope.parent_scopes)

    # Use the same name to denote sub-models
    pipeline_type = model.WhichOneof('Type')
    if pipeline_type == 'pipelineClassifier':
        sub_models = model.pipelineClassifier.pipeline.models
    elif pipeline_type == 'pipelineRegressor':
        sub_models = model.pipelineRegressor.pipeline.models
    elif pipeline_type == 'pipeline':
        sub_models = model.pipeline.models
    else:
        raise RuntimeError('Unsupported CoreML pipeline type: {0}'.format(pipeline_type))

    # Sequentially parse the sub-models
    # [TODO] Prepend '_' to all variable names but model I/O
    for sub_model in sub_models:
        # Declare the sub-model's input and output in this scope. Those input and output variables will be passed into
        # the sub-model's parsing function and connected with proper child variables.
        sub_inputs = []
        for var in sub_model.description.input:
            variable = scope.get_local_variable_or_declare_one(
                var.name,parse_coreml_feature_type(var.type, topology.default_batch_size))
            sub_inputs.append(variable)
        sub_outputs = []
        for var in sub_model.description.output:
            variable = scope.declare_local_variable(
                var.name,parse_coreml_feature_type(var.type, topology.default_batch_size))
            sub_outputs.append(variable)
        _parse_model(topology, scope, sub_model, sub_inputs, sub_outputs)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.input:
        # Find the first variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]
        # Create model's input variable. Note that we set prepend=True because model inputs should not hide any
        # intermediate variables.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature_type(var.type, topology.default_batch_size), prepend=True)
        # Feed the input to the sub-model's input. It's possible to add type conversion here by using a casting operator
        # rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.output:
        # Find the latest variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]
        # Create model's output variable
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature_type(var.type, topology.default_batch_size))
        # Connect the input and a sub-model's input. It's possible to add type conversion here by using a casting
        # operator rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_neural_network_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a neural network model.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the preprocessors and layers
        3. Connect model's (neither layers' nor preprocessors') inputs and outputs with proper variables created when
           parsing sub-models.
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML preprocessor/layer can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
        3. Preprocessor sometime may not include any information about its input
    '''

    # Create local scope to which all subsequent variables and operators belongs
    scope = topology.declare_scope('NeuralNetwork', [parent_scope] + parent_scope.parent_scopes)

    network = None
    network_type = model.WhichOneof('Type')
    if network_type == 'neuralNetworkClassifier':
        network = model.neuralNetworkClassifier
    elif network_type == 'neuralNetworkRegressor':
        network = model.neuralNetworkRegressor
    elif network_type == 'neuralNetwork':
        network = model.neuralNetwork
    else:
        raise RuntimeError('Unknown network type {}'.format(network_type))

    # [TODO] Prepend '_' to all variable names but model I/O
    for op in network.preprocessing:
        operator = scope.declare_local_operator(op.WhichOneof('preprocessor') + 'Preprocessor', op)

        # Infer the variable name to be processed if feature name is an empty string
        name = op.featureName if op.featureName != '' else model.description.input[0].name

        # Find out input variable
        original = scope.get_local_variable_or_declare_one(name)
        # [REVIEW] To avoid repeated assignments, we should pass type directly to get_local_variable_or_declare_one
        original.type = FloatTensorType()
        operator.inputs.append(original)

        # Declare a variable for storing the processed result
        processed = scope.declare_local_variable(name)
        # [REVIEW] To avoid repeated assignments, we should pass type directly to get_local_variable_or_declare_one
        processed.type = FloatTensorType()
        operator.outputs.append(processed)

    for op in network.layers:
        operator = scope.declare_local_operator(op.WhichOneof('layer'), op)

        # Find out input variable and connect them with the operator
        for name in op.input:
            variable = scope.get_local_variable_or_declare_one(name)
            # [REVIEW] To avoid repeated assignments, we should pass type directly to get_local_variable_or_declare_one

            # Although most neural network operators only accepts floats, we still need to handle the only exception
            # , embedding layer.
            if operator.type == 'embedding':
                variable.type = Int64TensorType()
            else:
                variable.type = FloatTensorType()
            operator.inputs.append(variable)

        # Declare variables for catching the operator's outputs
        for name in op.output:
            variable = scope.declare_local_variable(name)
            # [REVIEW] To avoid repeated assignments, we should pass type directly to get_local_variable_or_declare_one
            variable.type = FloatTensorType()
            operator.outputs.append(variable)

    sink_variables = scope.find_sink_variables()

    # Declare the model's inputs and outputs. Then, connect them with proper variables computed by the main network
    for var in model.description.input:
        # Search for the first variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]

        # Declare model input. To prevent intermediate variables form being hidden by model inputs, prepend is True.
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature_type(var.type, topology.default_batch_size), prepend=True)

        # A heuristic which forces the input of embedding to be integer tensor rather than float tensor.
        # Ideally this should be done by adding a cast operator, but ONNX doesn't have float-to-int casting.
        # If this variable is produced by another component in a CoreML pipeline, a bug may occur especially
        # when the source component's output type is float tensor.
        if isinstance(child_variable.type, Int64TensorType):
            variable.type = Int64TensorType(variable.type.shape)

        # Feed model input to the associated model input
        if variable.type != child_variable.type:
            operator_type = find_type_conversion(source_type=variable.type, target_type=child_variable.type)
        operator = scope.declare_local_operator(operator_type)
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)

    # Connect local input variables with proper variables from parent scope
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    for var in model.description.output:
        # CoreML's predicted label is not connected with any operator, so we handle it later as a special case.
        special_variable_names = [model.description.predictedFeatureName, model.description.predictedProbabilitiesName]
        if model.WhichOneof('Type') == 'neuralNetworkClassifier' and var.name in special_variable_names:
            continue
        # Search for the latest variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]

        # Create model output variable
        variable = scope.declare_local_variable(
            var.name, parse_coreml_feature_type(var.type, topology.default_batch_size))

        # Feed result calculated by the network to the output variable
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)

    # If predicted label exists, connect probability tensor and label by a special operator
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedFeatureName:
        # Find out the description of predicted label and declare a label variable
        label_variable = None
        for var in model.description.output:
            if var.name == model.description.predictedFeatureName:
                label_type = parse_coreml_feature_type(var.type, topology.default_batch_size)
                label_variable = scope.declare_local_variable(var.name, label_type)
                break
        operator = scope.declare_local_operator('tensorToLabel', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])
        operator.outputs.append(label_variable)

    # Probability tensor is implicitly converted into a dictionary (i.e., map) in CoreML. We handle this case here.
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedProbabilitiesName:
        operator = scope.declare_local_operator('tensorToProbabilityMap', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])

        # Find out the description of predicted probabilities and declare a variable for probability map
        for var in model.description.output:
            if var.name == model.description.predictedProbabilitiesName:
                probability_type = parse_coreml_feature_type(var.type, topology.default_batch_size)
                probability_variable = scope.declare_local_variable(var.name, probability_type)
                operator.outputs.append(probability_variable)
                break

    # Connect local output variables with proper variables from parent scope
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def parse_coreml(model, initial_types={}):
    '''
    This is the root function of the whole parsing procedure.
    :param model: CoreML model
    :param initial_types: a dictionary providing some types for some CoreML root variables
    :return: a Topology object. It's a intermediate representation of the input CoreML model

    Example of initial types:
    Assume that 'A' and 'B' are two root variable names used in a CoreML model. We can specify their types via
    >>> from _data_types import FloatTensorType
    >>> initial_type = {'A': FloatTensorType([40, 12, 1, 1]), 'B': FloatTensorType([1, 32, 1, 1])}
    '''

    # [TODO] Preserve top-level CoreML variable names
    reserved_variable_names = set()
    for var in list(model.description.input) + list(model.description.output):
        reserved_variable_names.add(var.name)
    default_batch_size = 1 if model.WhichOneof('Type') not in \
                              ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork'] else 'None'
    topology = Topology(model, default_batch_size, initial_types, reserved_variable_names)
    scope = topology.declare_scope('__root__')
    _parse_model(topology, scope, model)
    topology.compile()
    for variable in topology.find_root_and_sink_variables():
        if variable.raw_name not in reserved_variable_names:
            continue
        topology.rename_variable(variable.onnx_name, variable.raw_name)
    return topology
