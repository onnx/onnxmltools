import itertools
import numbers
import numpy as np
import six
from ._data_types import *
from ...proto import onnx_proto
from ...proto import helper


class ModelComponentContainer:
    '''
    This class is used to collect all materials required to build a ONNX GraphProto, which is usually encapsulated in
    ONNX ModelProto.
    '''
    def __init__(self):
        # Inputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.inputs = []
        # Outputs of ONNX graph. They are ValueInfoProto in ONNX.
        self.outputs = []
        # ONNX tensors (TensorProto). They are initializers of ONNX GraphProto.
        self.initializers = []
        # Intermediate variables of ONNX computational graph. They are ValueInfoProto in ONNX.
        self.value_info = []
        # ONNX NodeProto's used to define computation structure
        self.nodes = []

    def add_input(self, variable):
        '''
        Add our Variable object defined _parser.py into the the input list of the final ONNX model

        :param variable: The Variable object to be added
        '''
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        self.inputs.append(value_info)

    def add_output(self, variable):
        '''
        Add our Variable object defined _parser.py into the the output list of the final ONNX model

        :param variable: The Variable object to be added
        '''
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        self.outputs.append(value_info)

    def add_initializer(self, name, onnx_type, shape, content):
        '''
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        '''
        tensor = helper.make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)

    def add_value_info(self, variable):
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        self.value_info.append(value_info)

    def add_node(self, op_type, inputs, outputs, **attrs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        if not isinstance(inputs, list) or not all(isinstance(s, str) for s in inputs):
            raise ValueError('Inputs must be a list of string')
        if not isinstance(outputs, list) or not all(isinstance(s, str) for s in outputs):
            raise ValueError('Outputs must be a list of string')

        node = helper.make_node(op_type, inputs, outputs, **attrs)

        ml_domain = ["ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper", "DictVectorizer", "Imputer",
                     "FeatureVectorizer", "LabelEncoder", "LinearClassifier", "LinearRegressor", "Normalizer",
                     "OneHotEncoder", "Scaler", "SVMClassifier", "SVMRegressor", "TreeEnsembleClassifier",
                     "TreeEnsembleRegressor", "ZipMap"]
        if op_type in ml_domain:
            node.domain = 'ai.onnx.ml'

        self.nodes.append(node)


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
    for var in topology.raw_model.description.input:
        variable = next(variable for variable in model_inputs if variable.raw_name == var.name)
        container.add_input(variable)
    for var in topology.raw_model.description.output:
        variable = next(variable for variable in model_outputs if variable.raw_name == var.name)
        container.add_output(variable)

    # Traverse the graph from roots to leaves
    while not all(operator.is_evaluated for scope in topology.scopes for operator in scope.operators.values()):
        for scope in topology.scopes:
            for operator in scope.operators.values():
                if all(variable.is_fed for variable in operator.inputs) and not operator.is_evaluated:
                    # Similar to the evaluation of a operation, unknown inputs are not allowed
                    for variable in operator.inputs:
                        if not variable.is_fed:
                            raise RuntimeError('No input can be unknown')

                    # Convert the selected operator into some ONNX objects and save them into the container
                    converter_table[operator.type](scope, operator, container)

                    # Check if over-writing problem occurs (i.e., multiple operators produce results on one variable).
                    for variable in operator.outputs:
                        # Throw an error if this variable has been treated as an output somewhere
                        if variable.is_fed:
                            raise RuntimeError('One variable can only be assigned once')
                        # Mark this variable as filled
                        variable.is_fed = True

                    # Make this operator as handled
                    operator.is_evaluated = True

    graph = helper.make_graph(container.nodes, model_name, container.inputs, container.outputs, container.initializers)
    graph.value_info.extend(container.value_info)
    return helper.make_model(graph)


# [TODO] Check if this function is useful
def is_numeric_type(item):
    types = (six.integer_types)
    if (isinstance(item, list)):
        return isinstance(item[0], types)
    elif isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, int) or \
               np.issubdtype(item.dtype, float) or \
               isinstance(item.dtype, six.integer_types)
    else:
        return isinstance(item, types)


# [TODO] Check if this function is useful
def is_string_type(item):
    types = (six.string_types, six.text_type)

    if isinstance(item, list):
        return isinstance(item[0], types)
    if isinstance(item, np.ndarray):
        return isinstance(item.dtype, types)
    return isinstance(item, types)


# [TODO] Merge it into convert_to_list
def cast_list(type, items):
    return [type(item) for item in items]


def convert_to_python_value(var):
    if isinstance(var, numbers.Integral):
        return int(var)
    elif isinstance(var, numbers.Real):
        return float(var)
    elif isinstance(var, str):
        return str(var)
    else:
        raise TypeError('Unable to convert {0} to python type'.format(type(var)))


def convert_to_python_default_value(var):
    if isinstance(var, numbers.Integral):
        return int()
    elif isinstance(var, numbers.Real):
        return float()
    elif isinstance(var, str):
        return str()
    else:
        raise TypeError('Unable to find default python value for type {0}'.format(type(var)))


def convert_to_list(var):
    if isinstance(var, numbers.Real) or isinstance(var, str):
        return [convert_to_python_value(var)]
    elif isinstance(var, np.ndarray) and len(var.shape) == 1:
        return [convert_to_python_value(v) for v in var]
    elif isinstance(var, list):
        flattened = []
        if all(isinstance(ele, np.ndarray) and len(ele.shape) == 1 for ele in var):
            max_classes = max([ele.shape[0] for ele in var])
            flattened_one = []
            for ele in var:
                for i in range(max_classes):
                    if i < ele.shape[0]:
                        flattened_one.append(convert_to_python_value(ele[i]))
                    else:
                        flattened_one.append(convert_to_python_default_value(ele[0]))
            flattened += flattened_one
            return flattened
        elif all(isinstance(v, numbers.Real) or isinstance(v, str) for v in var):
            return [convert_to_python_value(v) for v in var]
        else:
            raise TypeError('Unable to flatten variable')
    else:
        raise TypeError('Unable to flatten variable')


def deduce_broadcast_axis_and_shape(shape):
    # This function is used to calculate the first axis aligned with the scalar and the scalar's ONNX shape for reduce-
    # like operators. Assuming input variable is always a 4-D tensor, we provide a few of examples. If scalar's shape
    # is [1, 2, 3] and input shape is [5, 2, 3, 8], the aligned axis is the [2] (indexed by 1 because indexes are 0-based)
    # in [5, 2, 3, 8], and the desired scalar shape in ONNX is [2, 3] # (the [1] in [1, 2, 3] is redundant and can cause
    # errors in ONNX's boardcasting). If the scaler's shape is [1], no matter what shape the input is, we leave the axis
    # "None" because ONNX operator may automatically handle it.

    # Input shape is [N, C, H, W]
    if len(shape) == 1:
        if shape[0] == 1:
            # shape is [1], we don't specify axis because it's a scalar
            return None, [1]
        else:
            # shape is [C], alignment starting at C-axis (indexed by 1)
            return 1, shape
    elif len(shape) == 3:
        if shape[0] == 1:
            # shape is [1, H, W], alignment starting at H-axis (indexed by 2)
            return 2, [shape[1], shape[2]]
        else:
            # shape is [C, H, W], alignment starting at C-axis (indexed by 1)
            return 1, shape


def extract_rnn_activation_info(activation):
    activation_type = activation.WhichOneof('NonlinearityType')
    alpha = None
    beta = None

    activation_map = {'linear': 'Affine',
                      'ReLU': 'Relu',
                      'leakyReLU': 'LeakyRelu',
                      'thresholdedReLU': 'ThresholdedRelu',
                      'PReLU': 'PRelu',
                      'tanh': 'Tanh',
                      'scaledTanh': 'ScaledTanh',
                      'sigmoid': 'Sigmoid',
                      'sigmoidHard': 'HardSigmoid',
                      'ELU': 'Elu',
                      'softsign': 'Softsign',
                      'softplus': 'Softplus',
                      'parametricSoftplus': 'ParametricSoftplus'}

    if activation_type not in activation_map:
        raise ValueError('Unsupported activation function: {}'.format(activation_type))

    # Notice that if we see a default vaulue (i.e., 0 for float), we may replace it with
    # the real default parameter for the specified activation function if necessary.
    if activation_type == 'leakyReLU':
        alpha = activation.leakyReLU.alpha
        if alpha == 0:
            alpha = 0.3
    elif activation_type == 'PReLU':
        raise RuntimeError('Unsupported activation function: {}'.format(activation_type))
    elif activation_type == 'ELU':
        alpha = activation.ELU.alpha
    elif activation_type == 'thresholdedReLU':
        alpha = activation.thresholdedReLU.alpha
        if alpha == 0:
            alpha = 1.0
    elif activation_type == 'scaledTanh':
        alpha = activation.scaledTanh.alpha
        beta = activation.scaledTanh.beta
    elif activation_type == 'linear':
        alpha = activation.linear.alpha
        beta = activation.linear.beta
        if alpha == 0:
            alpha = 1.0
    elif activation_type == 'sigmoidHard':
        alpha = activation.sigmoidHard.alpha
        beta = activation.sigmoidHard.beta
        if alpha == 0:
            alpha = 0.2
        if beta == 0:
            beta = 0.5
    elif activation_type == 'parametricSoftplus':
        raise RuntimeError('Unsupported activation function: {}'.format(activation_type))

    return activation_map[activation_type], alpha, beta


def convert_activation(scope, operator, container):
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    params = operator.raw_operator.activation
    activation_type = params.WhichOneof('NonlinearityType')
    if activation_type == 'leakyReLU':
        op_type = 'LeakyRelu'
        attrs['alpha'] = params.leakyReLU.alpha
    elif activation_type == 'ReLU':
        op_type = 'Relu'
    elif activation_type == 'PReLU':
        op_type = 'PRelu'
        attrs['slope'] = params.PReLU.alpha
    elif activation_type == 'ELU':
        op_type = 'Elu'
        attrs['alpha'] = params.ELU.alpha
    elif activation_type == 'thresholdedReLU':
        op_type = 'ThresholdedRelu'
        attrs['alpha'] = params.thresholdedReLU.alpha
    elif activation_type == 'tanh':
        op_type = 'Tanh'
    elif activation_type == 'scaledTanh':
        op_type = 'ScaledTanh'
        attrs['alpha'] = params.scaledTanh.alpha
        attrs['beta'] = params.scaledTanh.beta
    elif activation_type == 'linear':
        op_type = 'Affine'
        attrs['alpha'] = params.linear.alpha
        attrs['beta'] = params.linear.beta
    elif activation_type == 'sigmoid':
        op_type = 'Sigmoid'
    elif activation_type == 'sigmoidHard':
        op_type = 'HardSigmoid'
        attrs['alpha'] = params.sigmoidHard.alpha
        attrs['beta'] = params.sigmoidHard.beta
    elif activation_type == 'softsign':
        op_type = 'Softsign'
    elif activation_type == 'softplus':
        op_type = 'Softplus'
    elif activation_type == 'parametricSoftplus':
        op_type = 'ParametricSoftplus'
        attrs['alpha'] = params.parametricSoftplus.alpha
        attrs['beta'] = params.parametricSoftplus.beta
    else:
        raise TypeError('Unsupported activation layer {0}'.format(activation_type))

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_inner_product(scope, operator, container):
    params = operator.raw_operator.innerProduct
    op_type = 'FC'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    name_w = operator.full_name + '.W'
    shape_w = [params.outputChannels, params.inputChannels]
    inputs.append(name_w)
    container.add_initializer(name_w, onnx_proto.TensorProto.FLOAT, shape_w, params.weights.floatValue)

    if params.hasBias:
        name_b = operator.full_name + '.B'
        shape_b = [params.outputChannels]
        inputs.append(name_b)
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)

    attrs['axis'] = 1
    attrs['axis_w'] = 1

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_identity(scope, operator, container):
    op_type = 'Identity'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}
    container.add_node(op_type, inputs, outputs, **attrs)


def convert_softmax(scope, operator, container):
    op_type = 'Softmax'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}
    container.add_node(op_type, inputs, outputs, **attrs)


def convert_convolution(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import SamePadding

    params = operator.raw_operator.convolution
    op_type = 'ConvTranspose' if params.isDeconvolution else 'Conv'
    inputs = [operator.inputs[0].full_name]
    outputs = [operator.outputs[0].full_name]
    attrs = {'name': operator.full_name}

    shape_w = [params.outputChannels, int(params.kernelChannels / params.nGroups), params.kernelSize[0],
               params.kernelSize[1]]
    if params.isDeconvolution:
        shape_w[0], shape_w[1] = shape_w[1], shape_w[0]
    name_w = operator.full_name + '.W'
    inputs.append(name_w)
    container.add_initializer(name_w, onnx_proto.TensorProto.FLOAT, shape_w, params.weights.floatValue)

    if params.hasBias:
        shape_b = [len(params.bias.floatValue)]
        name_b = operator.full_name + '.B'
        inputs.append(name_b)
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)

    dilations = [1, 1]
    if len(params.dilationFactor) > 0:
        dilations = [params.dilationFactor[0], params.dilationFactor[1]]

    attrs['dilations'] = dilations
    attrs['group'] = params.nGroups
    attrs['kernel_shape'] = params.kernelSize

    pads = None
    auto_pad = None
    pad_type = params.WhichOneof('ConvolutionPaddingType')
    if pad_type == 'valid':

        if len(params.valid.paddingAmounts.borderAmounts) > 0:
            pads = [0, 0, 0, 0]
            pads[0] = params.valid.paddingAmounts.borderAmounts[0].startEdgeSize
            pads[1] = params.valid.paddingAmounts.borderAmounts[1].startEdgeSize
            pads[2] = params.valid.paddingAmounts.borderAmounts[0].endEdgeSize
            pads[3] = params.valid.paddingAmounts.borderAmounts[1].endEdgeSize
            # If padding amounts are all zero, there should be no padding list.
            if all(pad == 0 for pad in pads):
                pads = None
                auto_pad = 'VALID'
        else:
            auto_pad = 'VALID'

    elif pad_type == 'same':

        if params.same.asymmetryMode == SamePadding.BOTTOM_RIGHT_HEAVY:
            auto_pad = 'SAME_LOWER'
        elif params.same.asymmetryMode == SamePadding.TOP_LEFT_HEAVY:
            auto_pad = 'SAME_UPPER'
        else:
            raise ValueError('Unknown asymmetric mode: {}'.format(
                params.same.asymmetryMode))

    else:
        raise ValueError('Unsupported padding mode: {}'.format(pad_type))

    if params.isDeconvolution and len(params.outputShape) > 0:
        attrs['output_shape'] = params.outputShape

    if pads is not None:
        attrs['pads'] = pads

    if auto_pad is not None:
        attrs['auto_pad'] = auto_pad

    attrs['strides'] = params.stride

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_pooling(scope, operator, container):
    params = operator.raw_operator.pooling
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]

    from coremltools.proto.NeuralNetwork_pb2 import PoolingLayerParams as Params
    from coremltools.proto.NeuralNetwork_pb2 import SamePadding
    # Handle global pooling mode
    attrs = {'name': operator.full_name}
    if params.globalPooling:
        pooling_table = {params.MAX: 'GlobalMaxPool',
                         Params.AVERAGE: 'GlobalAveragePool',
                         Params.L2: 'GlobalLpPool'}

        if params.type not in pooling_table:
            raise ValueError('Unsupported pooling type: {}'.format(params.type))

        op_type = pooling_table[params.type]
        if params.type == Params.L2:
            attrs['p'] = 2

        container.add_node(op_type, inputs, outputs, **attrs)
        return

    # Handle local pooling mode
    if params.type == Params.MAX:
        op_type = 'MaxPool'
        attrs['dilations'] = [1, 1]
        attrs['kernel_shape'] = params.kernelSize
        attrs['strides'] = params.stride
    elif params.type == Params.AVERAGE:
        op_type = 'AveragePool'
        attrs['kernel_shape'] = params.kernelSize
        attrs['strides'] = params.stride
    elif params.type == Params.L2:
        op_type = 'LpPool'
        attrs['kernel_shape'] = params.kernelSize
        attrs['strides'] = params.stride
        attrs['p'] = 2
    else:
        raise ValueError('Unsupported pooling type: {}'.format(params.type))

    # Set up padding attributes
    pads = None
    auto_pad = None
    pad_type = params.WhichOneof('PoolingPaddingType')
    if pad_type == 'valid':

        if len(params.valid.paddingAmounts.borderAmounts) > 0:
            pads = [0, 0, 0, 0]
            pads[0] = params.valid.paddingAmounts.borderAmounts[0].startEdgeSize
            pads[1] = params.valid.paddingAmounts.borderAmounts[1].startEdgeSize
            pads[2] = params.valid.paddingAmounts.borderAmounts[0].endEdgeSize
            pads[3] = params.valid.paddingAmounts.borderAmounts[1].endEdgeSize
            # If padding amounts are all zero, there should be no padding list.
            if all(pad == 0 for pad in pads):
                pads = None
                auto_pad = 'VALID'
        else:
            auto_pad = 'VALID'

    elif pad_type == 'same':

        if params.same.asymmetryMode == SamePadding.BOTTOM_RIGHT_HEAVY:
            auto_pad = 'SAME_LOWER'
        elif params.same.asymmetryMode == SamePadding.TOP_LEFT_HEAVY:
            auto_pad = 'SAME_UPPER'
        else:
            raise ValueError('Unknown asymmetric mode: {}'.format(params.same.asymmetryMode))

    elif pad_type == 'includeLastPixel':

        # This padding mode is not officially supported in ONNX, so we use a
        # deprecated feature in Caffe. Notice that this is just a temporal
        # solution for unblocking some users.

        attrs['legacy_pad'] = 3
        pads = [0, 0, 0, 0]
        pads[0] = params.includeLastPixel.paddingAmounts[0]
        pads[1] = params.includeLastPixel.paddingAmounts[1]
        pads[2] = pads[0]
        pads[3] = pads[1]

    else:
        raise ValueError('Unsupported padding mode: {}'.format(pad_type))

    if pads is not None:
        attrs['pads'] = pads

    if auto_pad is not None:
        attrs['auto_pad'] = auto_pad
    # [TODO] Handle exclude_pad_area flag in CoreML's average pooling operator

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_preprocessing_scaler(scope, operator, container):
    params = operator.raw_operator.scaler
    attrs = {'name': operator.full_name, 'scale': params.channelScale}
    color_space = operator.inputs[0].type.color_space
    if color_space == 'GRAY':
        attrs['bias'] = [params.grayBias]
    elif color_space == 'RGB':
        attrs['bias'] = [params.redBias, params.greenBias, params.blueBias]
    elif color_space == 'BGR':
        attrs['bias'] = [params.blueBias, params.greenBias, params.redBias]
    else:
        raise ValueError('Unknown color space for tensor {}'.format(operator.inputs[0].full_name))

    container.add_node('ImageScaler', [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)


def convert_flatten(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import FlattenLayerParams as Params

    variable_to_be_flattened_name = operator.inputs[0].full_name
    flattened_variable_name = operator.outputs[0].full_name

    if operator.raw_operator.flatten.mode == Params.CHANNEL_LAST:
        op_type = 'Transpose'
        transpose_operator_name = scope.get_unique_operator_name(op_type)
        transpose_attrs = {'name': transpose_operator_name, 'perm': [0, 2, 3, 1]}
        transposed_variable_name = scope.get_unique_variable_name('transposed')

        container.add_node(op_type, [variable_to_be_flattened_name], [transposed_variable_name], **transpose_attrs)
        variable_to_be_flattened_name = transposed_variable_name

    op_type = 'Flatten'
    flatten_attrs = {'name': operator.full_name, 'axis': 1}

    container.add_node(op_type, [variable_to_be_flattened_name], [flattened_variable_name], **flatten_attrs)


def convert_perumte(scope, operator, container):
    op_type = 'Transpose'
    attrs = {'name': operator.full_name, 'parm': operator.raw_operator.permute.axis}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_image_to_float_tensor(scope, operator, container):
    op_type = 'Identity'
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_unidirectional_lstm(scope, operator, container):
    # The LSTM inputs are feature vector, X, initial hidden state, h_init, and initial cell state, c_init.
    # In CorML, their shapes respectively are [S, C_in], [1, C_out], and [1, C_out], where C_in is input feature
    # length, # C_out is output dimension, and S is sequence length. Note that S-axis is also known as time axis.
    # In ONNX, those shapes become [S, N, C_in] (X), [D, N, C_out] (h_init), and [D, N, C_out]. To simulate
    # CoreML LSTM under ONNX, we need some extra operators in addition to LSTM itself.
    #
    # Note that N=1 and D=1 are always true in ONNX if we are considering LSTM in CoreML because there is no
    # batch size in CoreML spec and CoreML LSTM is always uni-directional.
    #
    # Below we provide a visualization of our conversion for CoreML LSTM.
    #
    # Symbols:
    #
    #  X: input features of CoreML LSTM
    #  h_init: initial LSTM hidden state in CoreML
    #  c_init: initial LSTM cell state in CoreML
    #  Y: CoreML LSTM's output. It can be [S, C_out] (if sequence_output is on) or [1, C_out] (if sequence_output is off)
    #  Y_h: CoreML LSTM's last hidden state
    #  Y_c: CoreML LSTM's last cell state
    #
    #  X': input features of ONNX LSTM
    #  h_init': initial LSTM hidden state of ONNX
    #  c_init': initial LSTM hidden state of ONNX
    #  Y': ONNX LSTM's output
    #  Y_h': ONNX LSTM's last hidden state
    #  Y_c': ONNX LSTM's last cell state
    #
    # Computational graph of CoreML LSTM (if sequence_output is on):
    #
    #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
    #      |                 |                    |
    #      v                 v                    v
    #      '---------------  |    -----------------
    #                     |  |    |
    #                     v  v    v
    #                    CoreML LSTM
    #                     |  |    |
    #      ---------------'  l    '----------------
    #      |                 |                    |
    #      v                 v                    v
    #      Y [S, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
    #
    # Computational graph of CoreML LSTM in ONNX (if sequence_output is on):
    #
    #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
    #      |                 |                    |
    #      v                 v                    v
    #   Reshape           Reshape              Reshape
    #      |                 |                    |
    #      v                 v                    v
    #      X'[S, 1, C_in]  h_init'[1, 1, C_out] c_init [1, 1, C_out]
    #      |                 |                    |
    #      '---------------  |    -----------------
    #                     |  |    |
    #                     v  v    v
    #                     ONNX LSTM
    #                     |  |    |
    #       --------------'  vl   '---------------
    #      |                 |                    |
    #      v                 v                    v
    #      Y'[S, 1, C_out]  Y_h' [1, 1, C_out]   Y_c [1, 1, C_out]
    #      |                 |                    |
    #      v                 v                    v
    #   Reshape           Reshape              Reshape
    #      |                 |                    |
    #      v                 v                    v
    #      Y [S, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
    #
    # Computational graph of CoreML LSTM (if sequence_output is off):
    #
    #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
    #      |                 |                    |
    #      v                 v                    v
    #      '---------------  |    -----------------
    #                     |  |    |
    #                     v  v    v
    #                    CoreML LSTM
    #                     |  |    |
    #      ---------------'  l    '----------------
    #      |                 |                    |
    #      v                 v                    v
    #      Y [1, C_out]     Y_h [1, C_out]       Y_c [1, C_out]
    #           (Note that Y = Y_h)
    #
    # Computational graph of CoreML LSTM in ONNX (if sequence_output is off):
    #
    #      X [S, C_in]     h_init [1, C_out]    c_init [1, C_out]
    #      |                 |                    |
    #      v                 v                    v
    #   Reshape           Reshape              Reshape
    #      |                 |                    |
    #      v                 v                    v
    #      X'[S, 1, C_in]  h_init'[1, 1, C_out] c_init [1, 1, C_out]
    #      |                 |                    |
    #      '---------------  |    -----------------
    #                     |  |    |
    #                     v  v    v
    #                     ONNX LSTM
    #                     |  |    |
    #       --------------'  vl   '---------------
    #      |                 |                    |
    #      v                 v                    v
    #      Y'[S, 1, C_out]  Y_h' [1, 1, C_out]   Y_c [1, 1, C_out]
    #  (useless output)      |                    |
    #                        v                    v
    #                     Reshape              Reshape
    #                        |                    |
    #                        v                    v
    #                    Y [1, C_out]       Y_c [1, C_out]
    #                        |
    #                        v
    #                     Identity
    #                        |
    #                        v
    #                    Y_h [1, C_out]

    params = operator.raw_operator.uniDirectionalLSTM
    lstm_params = params.params
    lstm_weights = params.weightParams
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    # Initialize materials needed to create ONNX LSTM
    lstm_op_name = scope.get_unique_operator_name('LSTM')
    lstm_attrs = {'name': lstm_op_name}
    lstm_inputs = []
    lstm_outputs = []

    # Reshape input feature vector in CoreML format into ONNX format
    lstm_x_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_X_reshape')
    container.add_node('Reshape', operator.inputs[0].full_name, lstm_x_reshape_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1, 1, input_size])
    lstm_inputs.append(lstm_x_reshape_name)

    # Allocate LSTM's weight matrices and add them into ONNX LSTM's input list
    matrices_w = np.concatenate([lstm_weights.inputGateWeightMatrix.floatValue,
                                 lstm_weights.outputGateWeightMatrix.floatValue,
                                 lstm_weights.forgetGateWeightMatrix.floatValue,
                                 lstm_weights.blockInputWeightMatrix.floatValue])
    matrices_w_name = scope.get_unique_variable_name(lstm_op_name + '_W')
    container.add_initializer(matrices_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, input_size], matrices_w)
    lstm_inputs.append(matrices_w_name)

    # Allocate LSTM's recursion weight matrices and add them into ONNX LSTM's input list
    matrices_r = np.concatenate([lstm_weights.inputGateRecursionMatrix.floatValue,
                                 lstm_weights.outputGateRecursionMatrix.floatValue,
                                 lstm_weights.forgetGateRecursionMatrix.floatValue,
                                 lstm_weights.blockInputRecursionMatrix.floatValue])
    matrices_r_name = scope.get_unique_variable_name(lstm_op_name + '_R')
    container.add_initializer(matrices_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 4 * hidden_size, hidden_size], matrices_r)
    lstm_inputs.append(matrices_r_name)

    # Handle bias vectors
    vectors_b = np.zeros(shape=(8, hidden_size))
    if lstm_params.hasBiasVectors:
        vectors_b[0, :] = lstm_weights.inputGateBiasVector.floatValue
        vectors_b[1, :] = lstm_weights.outputGateBiasVector.floatValue
        vectors_b[2, :] = lstm_weights.forgetGateBiasVector.floatValue
        vectors_b[3, :] = lstm_weights.blockInputBiasVector.floatValue
    if lstm_params.forgetBias:
        # One may think we should do something like b[2, :] += 1., but it's wrong as CoreML has
        # added 1 into lstm_weights.forgetGateBiasVector.floatValue.
        pass
    if lstm_params.hasBiasVectors or lstm_params.forgetBias:
        vectors_b_name = scope.get_unique_variable_name(lstm_op_name + '_B')
        container.add_initializer(vectors_b_name, onnx_proto.TensorProto.FLOAT,
                                  [1, 8 * hidden_size], vectors_b.flatten())
        lstm_inputs.append(vectors_b_name)
    else:
        lstm_inputs.append('')

    # Converting CoreML LSTM doesn't need sequence length
    lstm_inputs.append('')

    # Provide ONNX LSTM the initial hidden state when necessary
    if len(operator.inputs) > 1:
        # Assign a Reshape to adjust CoreML hidden state's shape [1, C]/[1, C, 1, 1] into its ONNX counterpart [1, 1, C]
        lstm_h_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_h_init_reshape')
        container.add_node('Reshape', operator.inputs[1].full_name, lstm_h_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, 1, hidden_size])
        lstm_inputs.append(lstm_h_init_reshape_name)
        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
    else:
        lstm_inputs.append('')

    # Provide ONNX LSTM the initial cell state when necessary
    if len(operator.inputs) > 2:
        lstm_c_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_c_init_reshape')
        container.add_node('Reshape', operator.inputs[2].full_name, lstm_c_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, 1, hidden_size])
        lstm_inputs.append(lstm_c_init_reshape_name)
        # Add a zero initializer to initial cell state so that this variable becomes optional
        container.add_initializer(operator.inputs[2].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[2].type.shape,
                                  np.zeros(shape=operator.inputs[2].type.shape).flatten())
    else:
        lstm_inputs.append('')

    # Add peephole vector when presenting
    if lstm_params.hasPeepholeVectors:
        vectors_p = np.concatenate([lstm_weights.inputGatePeepholeVector.floatValue,
                                    lstm_weights.outputGatePeepholeVector.floatValue,
                                    lstm_weights.forgetGatePeepholeVector.floatValue])
        vectors_p_name = scope.get_unique_variable_name(lstm_op_name + '_P')
        container.add_initializer(vectors_p_name, onnx_proto.TensorProto.FLOAT,
                                  [1, 3 * hidden_size], vectors_p)
        lstm_inputs.append(vectors_p_name)
    else:
        lstm_inputs.append('')

    # Parse activation functions' information and add them into ONNX LSTM's attribute dictionary
    activation_types = []
    alphas = []
    betas = []
    for activation in params.activations:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    lstm_attrs['activations'] = activation_types
    if alphas:
        lstm_attrs['activation_alpha'] = alphas
    if betas:
        lstm_attrs['activation_beta'] = betas

    lstm_attrs['direction'] = 'reverse' if params.reverseInput else 'forward'
    lstm_attrs['output_sequence'] = lstm_params.sequenceOutput
    lstm_attrs['hidden_size'] = hidden_size
    lstm_attrs['clip'] = lstm_params.cellClipThreshold
    lstm_attrs['input_forget'] = lstm_params.coupledInputAndForgetGate

    # Handle the first output of LSTM
    if lstm_params.sequenceOutput:
        # Handle the first output of LSTM
        lstm_y_name = scope.get_unique_variable_name(lstm_op_name + '_Y')
        lstm_outputs.append(lstm_y_name)
        container.add_node('Reshape', lstm_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, hidden_size])

        # Handle the second output of LSTM
        if len(operator.outputs) > 1:
            lstm_y_h_name = scope.get_unique_variable_name(lstm_op_name + '_Y_h')
            lstm_outputs.append(lstm_y_h_name)
            container.add_node('Reshape', lstm_y_h_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        lstm_outputs.append(scope.get_unique_variable_name('isolated'))

        # Use the second output of ONNX LSTM to produce the first output of CoreML LSTM
        lstm_y_name = scope.get_unique_variable_name(lstm_op_name + '_Y')
        lstm_outputs.append(lstm_y_name)
        container.add_node('Reshape', lstm_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])

        # Create the second LSTM output from the first output
        if len(operator.outputs) > 1:
            container.add_node('Identity', operator.outputs[0].full_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Identity'))

    # Handle the cell state output of LSTM
    if len(operator.outputs) > 2:
        lstm_c_name = scope.get_unique_variable_name(lstm_op_name + '_Y_c')
        lstm_outputs.append(lstm_c_name)
        container.add_node('Reshape', lstm_c_name, operator.outputs[2].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])

    # Finally, the main LSTM operator is created
    container.add_node('LSTM', lstm_inputs, lstm_outputs, **lstm_attrs)


def convert_embedding(scope, operator, container):
    params = operator.raw_operator.embedding
    gather_op_name = scope.get_unique_operator_name('Gather')
    gather_attrs = {'name': gather_op_name}

    # Reshape the indexes we want to embed to 1-D tensor. Otherwise, ONNX Gather's output may get wrong shape.
    reshaped_input_name = scope.get_unique_variable_name(gather_op_name + 'input_reshaped')  # 2nd input of Gather
    container.add_node('Reshape', operator.inputs[0].full_name, reshaped_input_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1])

    # Load the embedding matrix. Its shape is outputChannels-by-inputDim.
    weights = np.array(params.weights.floatValue).reshape(params.outputChannels, params.inputDim)
    weights_name = scope.get_unique_variable_name(gather_op_name + '_W')  # 1st input of Gather
    container.add_initializer(weights_name, onnx_proto.TensorProto.FLOAT,
                              [params.inputDim, params.outputChannels], weights.transpose().flatten().tolist())

    # To support the bias term in an embedding (if exists), we need to create one extra node.
    if params.hasBias:
        # Put the embedded result onto a temporal tensor
        gather_output_name = scope.get_unique_variable_name(gather_op_name + '_output')
        container.add_node('Gather', [weights_name, reshaped_input_name], gather_output_name, **gather_attrs)

        # Load the bias vector into an initializer
        bias_name = scope.get_unique_variable_name(gather_op_name + '_bias')
        container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT,
                                  [params.outputChannels], params.bias.floatValue)
        # Create an addition operator to add bias (shape: [C]) into Gather's tensor (shape: [N, C])
        container.add_node('Add', [gather_output_name, bias_name], operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Add'), axis=1, broadcast=1)
    else:
        # There is no bias, so we just output the result produced by the embedding node.
        container.add_node('Gather', [weights_name, reshaped_input_name], operator.output_full_names, **gather_attrs)


def convert_concat(scope, operator, container):
    op_type = 'Concat'
    attrs = {'name': operator.full_name}
    if operator.raw_operator.concat.sequenceConcat:
        attrs['axis'] = 0
    else:
        attrs['axis'] = 1

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_permute(scope, operator, container):
    op_type = 'Transpose'
    attrs = {'name': operator.full_name}
    attrs['perm'] = operator.raw_operator.permute.axis

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_reshape(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReshapeLayerParams as Params

    params = operator.raw_operator.reshape

    if params.mode == Params.CHANNEL_LAST:
        op_type = 'Transpose'
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_transpose')
        attrs = {'name': scope.get_unique_operator_name(op_type), 'perm': [0, 2, 3, 1]}
        container.add_node(op_type, [operator.inputs[0].full_name], [intra_variable_name], **attrs)
    else:
        intra_variable_name = operator.inputs[0].full_name

    op_type = 'Reshape'
    attrs = {'name': operator.full_name, 'shape': params.targetShape}
    container.add_node(op_type, [intra_variable_name], [operator.outputs[0].full_name], **attrs)


def convert_tensor_to_label(scope, operator, container):
    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = [s.encode('ascii') for s in model.stringClassLabels.vector]
            label_type = onnx_proto.TensorProto.STRING
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            labels = model.int64ClassLabels.vector
            label_type = onnx_proto.TensorProto.INT64
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            labels = [s.encode('ascii') for s in model.pipelineClassifier.stringClassLabels.vector]
            label_type = onnx_proto.TensorProto.STRING
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            labels = model.int64ClassLabels.vector
            label_type = onnx_proto.TensorProto.INT64
        else:
            raise ValueError('Unknown label type found')
    else:
        raise TypeError('Only neural network classifiers and pipeline classifiers are supported')

    # Use a Constant operator to load and output all labels as a tensor
    label_loader_name = scope.get_unique_operator_name('LabelLoader')
    label_loader_attrs = {'name': label_loader_name}
    label_buffer_name = scope.get_unique_variable_name('ClassLabels')
    label_loader_attrs['value'] = helper.make_tensor(label_buffer_name, label_type, [len(labels)], labels)
    container.add_node('Constant', [], [label_buffer_name], **label_loader_attrs)

    # Extract most possible label index
    label_id_extractor_name = scope.get_unique_operator_name('LabelIndexExtractor')
    label_id_extractor_attrs = {'name': label_id_extractor_name}
    # [TODO] We assume probability tensor's shape is [N, C]. Need to consider [C] cases.
    label_id_extractor_attrs['axis'] = 1
    label_id_extractor_attrs['keepdims'] = 1
    extracted_id_name = scope.get_unique_variable_name('LabelId')
    container.add_node('ArgMax', [operator.inputs[0].full_name], [extracted_id_name], **label_id_extractor_attrs)

    # Pick up the label indicated by the selected ID
    label_selector_name = scope.get_unique_operator_name('LabelSelector')
    label_selector_attrs = {'name': label_selector_name}
    # [TODO] Check if AFE can handle [N, C]
    container.add_node('ArrayFeatureExtractor', [label_buffer_name, extracted_id_name], [operator.outputs[0].full_name],
                       **label_selector_attrs)


def convert_dot(scope, operator, container):
    if operator.raw_operator.dot.cosineSimilarity:
        intra_variable_name1 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_normalized')
        normalizer_name1 = scope.get_unique_operator_name('L2NormNormalizer')
        attrs1 = {'name': normalizer_name1, 'p': 2., 'aixs': 1}
        container.add_node('LpNormalization', [operator.inputs[0].full_name], [intra_variable_name1], **attrs1)

        intra_variable_name2 = scope.get_unique_variable_name(operator.inputs[1].full_name + '_normalized')
        normalizer_name2 = scope.get_unique_operator_name('L2NormNormalizer')
        attrs2 = {'name': normalizer_name2, 'p': 2., 'aixs': 1}
        container.add_node('LpNormalization', [operator.inputs[1].full_name], [intra_variable_name2], **attrs2)
    else:
        intra_variable_name1 = operator.inputs[0].full_name
        intra_variable_name2 = operator.inputs[1].full_name

    product_name = scope.get_unique_variable_name(intra_variable_name1 + '_multiply_' + intra_variable_name2)
    multiplier_name = scope.get_unique_operator_name('Mul')
    product_attrs = {'name': multiplier_name}
    container.add_node('Mul', [intra_variable_name1, intra_variable_name2], [product_name], **product_attrs)

    reducer_name = scope.get_unique_operator_name('ReduceSum')
    reducer_attrs = {'name': reducer_name, 'axes': [1], 'keepdims': 0}
    container.add_node('ReduceSum', [product_name], operator.output_full_names, **reducer_attrs)


def convert_gru(scope, operator, container):
    # The GRU inputs are feature vector, X, and initial hidden state, h_init. Let C_in and C_out denote
    # the input feature length and the output dimension, respectively. Assume that S is the sequence length
    # (i.e., S-axis is the time axis of a sequence). In CorML, the shapes of X and h_init are [S, C_in] and
    # [1, C_out] respectively. In ONNX, the two shapes become [S, N, C_in] (X) and [D, N, C_out]
    # (h_init), where N is the batch size and S is the sequence length (i.e., S-axis is the time axis). To
    # simulate CoreML GRU under ONNX, we need to introduce some extra operators. Note that N=1 and D=1 always
    # hold in ONNX if we are considering GRU from CoreML because there is no batch size in CoreML and CoreML GRU
    # is always uni-directional.
    #
    # Below we provide a visualization of our conversion for CoreML GRU.
    #
    # Symbols:
    #
    #  X: input feature vector of CoreML GRU
    #  h_init: initial GRU state of CoreML
    #  Y: CoreML GRU's output. It can be [S, C_out] (if sequence_output is on ) or [1, C_out] (if sequence_output is off)
    #  Y_h: CoreML GRU's last hidden state
    #
    #  X': input features of ONNX GRU
    #  h_init': initial GRU state of ONNX
    #  Y': ONNX GRU's output
    #  Y_h': ONNX GRU's last hidden state
    #
    # Computational graph of CoreML GRU (sequence_output is on):
    #
    # X [S, C_in] ---> CoreML GRU ---> Y [S, C_out]
    #                    ^     |
    #                    |     |
    # h_init [1, C_cou] -'     '---> Y_h [1, C_out]
    #
    # Computational graph we use to repreent CoreML GRU into ONNX (sequence_output is on):
    #
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX GRU --> Y' [S, 1, C_out] --> Reshape --> Y [S, C_out]
    #                                                    ^ |
    #                                                    | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y_h [1, C_out]
    #
    # Computational graph of CoreML GRU (sequence_output is off):
    #
    # X [S, C_in] ---> CoreML GRU ---> Y [1, C_out]
    #                    ^     |
    #                    |     |
    # h_init [1, C_cou] -'     '---> Y_h [1, C_out] Note that in this case, Y=Y_h.
    #
    # Computational graph we use to represent CoreML GRU into ONNX (sequence_output is off):
    #
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] --> ONNX GRU --> Y' [S, 1, C_out] (this output won't be connected with others)
    #                                                    ^ |
    #                                                    | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
    #                                                                                                  |
    #                                                                                                  v
    #                                                                                               Identity
    #                                                                                                  |
    #                                                                                                  v
    #                                                                                             Y_h [1, C_out]

    params = operator.raw_operator.gru
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    # Initialize GRU's attributes. They will be used to build GRU in the end of this function.
    gru_op_name = scope.get_unique_operator_name('GRU')
    gru_attrs = {'name': gru_op_name}
    gru_inputs = []
    gru_outputs = []

    # Resahpe CoreML variable into ONNX format for feeding it into ONNX GRU
    gru_x_reshape_name = scope.get_unique_variable_name(gru_op_name + '_X_reshape')
    container.add_node('Reshape', operator.inputs[0].full_name, gru_x_reshape_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1, 1, input_size])
    gru_inputs.append(gru_x_reshape_name)

    # Create weight matrices of GRU and add it into ONNX GRU's input list
    matrices_w = np.concatenate([params.updateGateWeightMatrix.floatValue,
                                 params.resetGateWeightMatrix.floatValue,
                                 params.outputGateWeightMatrix.floatValue])
    matrices_w_name = scope.get_unique_variable_name(gru_op_name + '_W')
    container.add_initializer(matrices_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, input_size], matrices_w)
    gru_inputs.append(matrices_w_name)

    # Create recursion matrices of GRU and add it into ONNX GRU's input list
    matrices_r = np.concatenate([params.updateGateRecursionMatrix.floatValue,
                                 params.resetGateRecursionMatrix.floatValue,
                                 params.outputGateRecursionMatrix.floatValue])
    matrices_r_name = scope.get_unique_variable_name(gru_op_name + '_R')
    container.add_initializer(matrices_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, 3 * hidden_size, hidden_size], matrices_r)
    gru_inputs.append(matrices_r_name)

    if params.hasBiasVectors:
        # Create bias vectors of GRU and add them into ONNX GRU's input list
        vectors_b = np.concatenate([params.updateGateBiasVector.floatValue,
                                    params.resetGateBiasVector.floatValue,
                                    params.outputGateBiasVector.floatValue,
                                    np.zeros(3 * hidden_size)])
        vectors_b_name = scope.get_unique_variable_name(gru_op_name + '_B')
        container.add_initializer(vectors_b_name, onnx_proto.TensorProto.FLOAT,
                                  [1, 6 * hidden_size], vectors_b)
        gru_inputs.append(vectors_b_name)
    else:
        # Because operator's arguments are position-sensitive, we need an empty string even if
        # this variable doesn't exist.
        gru_inputs.append('')

    # The argument, sequence length, is always missing when converting CoreML GRU.
    gru_inputs.append('')

    # Handle initial hidden state if it exists
    if len(operator.inputs) == 2:
        # Change the shape of initial state in CoreML so that ONNX's GRU is willing to take it.
        gru_h_init_reshape_name = scope.get_unique_variable_name(gru_op_name + '_h_init')
        container.add_node('Reshape', operator.inputs[1].full_name, gru_h_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, 1, hidden_size])
        gru_inputs.append(gru_h_init_reshape_name)
        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
    else:
        # Because operator's arguments are position-sensitive, we need an empty string even if
        # this variable doesn't exist.
        gru_inputs.append('')

    activation_types = []
    alphas = []
    betas = []
    for activation in params.activations:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    gru_attrs['activations'] = activation_types
    if alphas:
        gru_attrs['activation_alpha'] = alphas
    if betas:
        gru_attrs['activation_beta'] = betas
    gru_attrs['direction'] = 'reverse' if params.reverseInput else 'forward'
    gru_attrs['output_sequence'] = params.sequenceOutput
    gru_attrs['hidden_size'] = hidden_size

    if params.sequenceOutput:
        # Again, the output shapes in ONNX's GRU is not consistent with that in CoreML, so we need
        # to adjust the result produced by ONNX according to CoreML format.
        gru_y_name = scope.get_unique_variable_name(gru_op_name + '_Y')
        gru_outputs.append(gru_y_name)
        container.add_node('Reshape', gru_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, hidden_size])

        # Handle the second output, the last hidden state of a sequence, if exists.
        if len(operator.outputs) == 2:
            gru_y_h_name = scope.get_unique_variable_name(gru_op_name + '_Y_h')
            gru_outputs.append(gru_y_h_name)
            container.add_node('Reshape', gru_y_h_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])
    else:
        # Recall that when sequence output is false, the first and the second outputs of GRU
        # are identical. Thus, we can ignore ONNX GRU's first output.
        gru_outputs.append(scope.get_unique_variable_name('isloated'))

        # As the two outputs are always identical, so we just need to compute one of them and
        # produce the other using identiy operator.
        gru_y_name = scope.get_unique_variable_name(gru_op_name + '_Y')
        gru_outputs.append(gru_y_name)

        container.add_node('Reshape', gru_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])

        if len(operator.outputs) == 2:
            container.add_node('Identity', operator.outputs[0].full_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Identity'))

    # Finally, we create the major GRU operator in ONNX.
    container.add_node('GRU', gru_inputs, gru_outputs, **gru_attrs)


def convert_l2_normalization(scope, operator, container):
    container.add_node('L2Normalization', operator.input_full_names,
                       operator.output_full_names, name=operator.full_name)


def convert_load_constant(scope, operator, container):
    params = operator.raw_operator.loadConstant
    constant_name = scope.get_unique_variable_name('constant')
    constant = helper.make_tensor(constant_name, onnx_proto.TensorProto.FLOAT, params.shape, params.data.floatValue)
    attrs = {'name': operator.full_name, 'value': constant}
    container.add_node('Constant', operator.input_full_names, operator.output_full_names, **attrs)


def convert_lrn(scope, operator, container):
    op_type = 'LRN'
    params = operator.raw_operator.lrn
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['size'] = params.localSize
    attrs['alpha'] = params.alpha
    attrs['beta'] = params.beta
    attrs['bias'] = params.k
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_max(scope, operator, container):
    op_type = 'Max'
    op_name = scope.get_unique_operator_name(op_type)
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)


def convert_preprocessing_mean_image(scope, operator, container):
    op_type = 'MeanSubtraction'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name, 'image': operator.raw_operator.meanImage}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_min(scope, operator, container):
    op_type = 'Min'
    op_name = scope.get_unique_operator_name(op_type)
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)


def convert_mean_variance_normalization(scope, operator, container):
    op_type = 'MeanVarianceNormalization'
    op_name = scope.get_unique_operator_name(op_type)
    params = operator.raw_operator.mvn
    attrs = {'name': op_name}
    attrs['across_channels'] = params.acrossChannels
    attrs['normalize_variance'] = params.normalizeVariance
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_multiply(scope, operator, container):
    op_type = 'Mul'
    op_name = scope.get_unique_operator_name(op_type)
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)


def convert_reduce(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params
    reduce_mode_table = {Params.SUM: 'ReduceSum', Params.AVG: 'ReduceMean', Params.PROD: 'ReduceProd',
                         Params.LOGSUM: 'ReduceLogSum', Params.SUMSQUARE: 'ReduceSumSquare',
                         Params.L1: 'ReduceL1', Params.L2: 'ReduceL2', Params.MAX: 'ReduceMax',
                         Params.MIN: 'ReduceMin', Params.ARGMAX: 'ArgMax'}

    params = operator.raw_operator.reduce
    reduce_mode = reduce_mode_table[params.mode]
    reduce_name = scope.get_unique_operator_name(reduce_mode)
    reduce_axis_table = {Params.CHW: [1, 2, 3], Params.HW: [2, 3], Params.C: [1], Params.H: [2], Params.W: [3]}
    reduce_axis = reduce_axis_table[params.axis]
    attrs = {'name': reduce_name, 'axes': reduce_axis}
    container.add_node(reduce_mode, operator.input_full_names, operator.output_full_names, **attrs)


def convert_reorganize_data(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params
    params = operator.raw_operator.reorganizeData
    if params.mode == Params.DEPTH_TO_SPACE:
        op_type = 'BatchToSpace'
    elif params.mode == Params.SPACE_TO_DEPTH:
        op_type = 'SpaceToBatch'
    else:
        raise ValueError('Unsupport reorganization mode {0}'.format(params.mode))

    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name, 'blocksize': params.blockSize}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_sequence_repeat(scope, operator, container):
    op_type = 'Tile'
    attrs = {'name': operator.full_name}
    attrs['tiles'] = operator.raw_operator.sequenceRepeat.nRepetitions
    attrs['axis'] = 0

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_slice(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params

    op_type = 'Slice'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name}
    params = operator.raw_operator.slice

    axis_map = {Params.CHANNEL_AXIS: 0, Params.HEIGHT_AXIS: 1, Params.WIDTH_AXIS: 2}
    starts = [0, 0, 0]
    ends = [-1, -1, -1]

    starts[axis_map[params.axis]] = params.startIndex
    ends[axis_map[params.axis]] = params.endIndex

    attrs['starts'] = starts
    attrs['ends'] = ends
    attrs['stride'] = params.stride

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_split(scope, operator, container):
    op_type = 'Split'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name, 'split': operator.raw_operator.split.nOutputs, 'axis': 1}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_unary(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UnaryFunctionLayerParams as Params

    params = operator.raw_operator.unary
    preprocessor_type = 'Affine'
    preprocessor_name = scope.get_unique_operator_name(preprocessor_type)
    preprocessor_attrs = {'name': preprocessor_name, 'alpha': params.scale, 'beta': params.shift}

    preprocessed_variable_name = scope.get_unique_variable_name(preprocessor_name + '_output')
    container.add_node(preprocessor_type, operator.input_full_names, [preprocessed_variable_name], **preprocessor_attrs)

    simple_unary_map = {Params.SQRT: 'Sqrt', Params.INVERSE: 'Reciprocal',
                        Params.EXP: 'Exp', Params.LOG: 'Log', Params.ABS: 'Abs'}

    if params.type == Params.RSQRT:
        op_type = 'Sqrt'
        sqrt_op_name = scope.get_unique_operator_name(op_type)
        sqrt_name = scope.get_unique_variable_name(op_type + '_output')
        container.add_node(op_type, [preprocessed_variable_name], [sqrt_name], name=sqrt_op_name)

        op_type = 'Reciprocal'
        inverse_op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [sqrt_name], operator.output_full_names, name=inverse_op_name)
    elif params.type == Params.POWER:
        exp_name = scope.get_unique_variable_name('Y')
        container.add_initializer(exp_name, onnx_proto.TensorProto.FLOAT, [1], [params.alpha])

        op_type = 'Pow'
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [operator.inputs[0].full_name, exp_name], operator.output_full_names, name=op_name)
    elif params.type == Params.THRESHOLD:
        op_type = 'Clip'
        op_name = scope.get_unique_operator_name(op_type)
        attrs = {'name': op_name, 'max': params.alpha}
        container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)
    elif params.type in simple_unary_map:
        op_type = simple_unary_map[params.type]
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)
    else:
        raise ValueError('Unsupported unary function :{}'.format(params.type))


def convert_upsample(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UpsampleLayerParams as Params
    op_type = 'Upsample'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    params = operator.raw_operator.upsample
    if params.mode == Params.NN:
        attrs['mode'] = 'NEAREST'
    elif params.mode == Params.BILINEAR:
        attrs['mode'] = 'BILINEAR'
    else:
        raise ValueError('Unsupported interpolation model in up-sampling')
    attrs['width_scale'] = float(params.scalingFactor[1])
    attrs['height_scale'] = float(params.scalingFactor[0])
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_bidirectional_lstm(scope, operator, container):
    # The LSTM inputs are feature vector (X), initial forward hidden state (h_init), initial backward hidden
    # state (h_init_rev), initial forward cell state (c_init), and initial backward cell stat (c_init_rev)e.
    # Because of the shape differences between ONNX LSTM and CoreML LSTM, the bidirectional LSTM conversion
    # is not straightforward. See some visualizations below for details.
    #
    # Symbols:
    #
    #  C_in: input feature length
    #  C_out: output feature length
    #  S: sequence length. For example, it can be the number of tokens in a sentence.
    #
    #  X: input features of CoreML LSTM. Its shape is [S, C_in].
    #  h_init: initial forward hidden state in CoreML. Its shape is [1, C_out].
    #  c_init: initial forward cell state in CoreML. Its shape is [1, C_out].
    #  h_init_rev: initial backward hidden state in CoreML. Its shape is [1, C_out].
    #  c_init_rev: initial backward cell state in CoreML. Its shape is [1, C_out].
    #  Y: CoreML LSTM's output. It can be [S, C_out] (if sequence_output is on) or [1, C_out] (if sequence_output is off)
    #  Y_h: CoreML LSTM's last foward hidden state. Its shape is [1, C_out].
    #  Y_c: CoreML LSTM's last forward cell state. Its shape is [1, C_out].
    #  Y_h_rev: CoreML LSTM's last backward hidden state. Its shape is [1, C_out].
    #  Y_c_rev: CoreML LSTM's last backward cell state. Its shape is [1, C_out].
    #
    #  X': input features of ONNX LSTM
    #  h_init': initial (forward and backward) hidden states of ONNX LSTM
    #  c_init': initial (forward and backward) hidden states of ONNX LSTM
    #  Y': ONNX LSTM's output
    #  Y_h': ONNX LSTM's last (forward and backward) hidden states
    #  Y_c': ONNX LSTM's last (forward and backward) cell states
    #
    # Computational graph of CoreML bi-directional LSTM (if sequence_output is on):
    #
    #      X [S, C_in]  h_init [1, C_out]  c_init [1, C_out]  h_init_rev [1, C_out]  c_init_rev [1, C_out]
    #      |               |                    |                |                      |
    #      |               '-----------------.  |                |                      |
    #      |                                 |  |                |   .------------------'
    #      '--------------------------------.|  |                |   |
    #                                       ||  |                |   |
    #                                       vv  v                v   v
    #                                       CoreML Bi-directional LSTM
    #                                       ||  |                |   |
    #      .--------------------------------'|  |                |   '------------------.
    #      |                                 |  |                |                      |
    #      |              .------------------'  |                |                      |
    #      |              |                     |                |                      |
    #      v              v                     v                v                      v
    #      Y [S, 2*C_out] Y_h [1, C_out]     Y_c [1, C_out]    Y_h_rev [1, C_out]     Y_c_rev [1, C_out]
    #
    # Computational graph of CoreML bi-directional LSTM in ONNX (if sequence_output is on):
    #
    #      X [S, C_in]  h_init [1, C_out]  h_init_rev [1, C_out] c_init [1, C_out]   c_init_rev [1, C_out]
    #      |               |                    |                  |                      |
    #      |               |                    |                  |                      |
    #      |               '-------.    .-------'                  '--------.    .--------'
    #      |                       |    |                                   |    |
    #      v                       v    v                                   v    v
    #   Reshape                    Concate                                  Concate
    #      |                          |                                        |
    #      v                          v                                        v
    #      X' [S, 1, C_in]        _h_init_ [2, C_out]                      _c_init_ [2, C_out]
    #      |                          |                                        |
    #      |                          v                                        v
    #      |                       Reshape                                  Reshape
    #      |                          |                                        |
    #      |                          v                                        V
    #      |                       h_init' [2, 1, C_out]                    c_init' [2, 1, Cout]
    #      |                          |                                        |
    #      |                          '-----------------.       .--------------'
    #      '----------------------------------.         |       |
    #                                         |         |       |
    #                                         v         v       v
    #                                       ONNX Bi-directional LSTM
    #                                         |         |       |
    #      .----------------------------------'         |       '--------------------.
    #      |                                            |                            |
    #      v                                            v                            v
    #      Y' [S, 2, 1, C_out]                           Y_h' [2, 1, C_out]           Y_c' [2, 1, C_out]
    #      |                                            |                            |
    #      v                                            v                            v
    #   Reshape                                      Reshape                      Reshape
    #      |                                            |                            |
    #      v                                            v                            v
    #      Y  [S, 2*C_out]                             _Y_h_' [2, C_out]            _Y_c_' [2, C_out]
    #                                                   |                            |
    #                                                   v                            v
    #                                                 Split                        Split
    #                                                 |   |                        |   |
    #                     .---------------------------'   |      .-----------------'   |
    #                     |                     .---------'      |                     |
    #                     |                     |                |                     |
    #                     v                     v                v                     v
    #                    Y_h [1, C_out]     Y_h_rev [1, C_out]  Y_c [1, C_out]       Y_c_rev [1, C_out]
    #
    # Computational graph of CoreML bi-directional LSTM (if sequence_output is off):
    #
    #      X [S, C_in]  h_init [1, C_out]  c_init [1, C_out]  h_init_rev [1, C_out]  c_init_rev [1, C_out]
    #      |               |                    |                |                      |
    #      |               '-----------------.  |                |                      |
    #      |                                 |  |                |   .------------------'
    #      '--------------------------------.|  |                |   |
    #                                       ||  |                |   |
    #                                       vv  v                v   v
    #                                       CoreML Bi-directional LSTM
    #                                       ||  |                |   |
    #      .--------------------------------'|  |                |   '------------------.
    #      |                                 |  |                |                      |
    #      |              .------------------'  |                |                      |
    #      |              |                     |                |                      |
    #      v              v                     v                v                      v
    #      Y [1, 2*C_out] Y_h [1, C_out]     Y_c [1, C_out]    Y_h_rev [1, C_out]     Y_c_rev [1, C_out]
    #
    # Computational graph of CoreML bi-directional LSTM in ONNX (if sequence_output is off):
    #
    #      X [S, C_in]  h_init [1, C_out]  h_init_rev [1, C_out] c_init [1, C_out]   c_init_rev [1, C_out]
    #      |               |                    |                  |                      |
    #      |               |                    |                  |                      |
    #      |               '-------.    .-------'                  '--------.    .--------'
    #      |                       |    |                                   |    |
    #      v                       v    v                                   v    v
    #   Reshape                    Concate                                  Concate
    #      |                          |                                        |
    #      v                          v                                        v
    #      X' [S, 1, C_in]        _h_init_ [2, C_out]                      _c_init_ [2, C_out]
    #      |                          |                                        |
    #      |                          v                                        v
    #      |                       Reshape                                  Reshape
    #      |                          |                                        |
    #      |                          v                                        V
    #      |                       h_init' [2, 1, C_out]                    c_init' [2, 1, Cout]
    #      |                          |                                        |
    #      |                          '-----------------.       .--------------'
    #      '----------------------------------.         |       |
    #                                         |         |       |
    #                                         v         v       v
    #                                       ONNX Bi-directional LSTM
    #                                         |         |       |
    #      .----------------------------------'         |       '--------------------.
    #      |                                            |                            |
    #      v                                            v                            v
    #      Y' [S, 2, 1, C_in]          .-------------- Y_h' [2, 1, C_out]           Y_c' [2, 1, C_out]
    #   (useless output)               |                |                            |
    #                                  v                v                            v
    #                               Reshape          Reshape                      Reshape
    #                                  |                |                            |
    #                                  |                v                            v
    #   .------------------------------'              _Y_h_' [2, C_out]            _Y_c_' [2, C_out]
    #   |                                               |                            |
    #   |                                               v                            v
    #   |                                             Split                        Split
    #   |                                             |   |                        |   |
    #   |                 .---------------------------'   |      .-----------------'   |
    #   |                 |                     .---------'      |                     |
    #   |                 |                     |                |                     |
    #   v                 v                     v                v                     v
    #   Y  [1, 2*C_out]   Y_h [1, C_out]     Y_h_rev [1, C_out]  Y_c [1, C_out]       Y_c_rev [1, C_out]

    params = operator.raw_operator.biDirectionalLSTM
    lstm_params = params.params
    lstm_weights = params.weightParams
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    lstm_op_name = scope.get_unique_operator_name('LSTM')
    lstm_attrs = {'name': lstm_op_name}
    lstm_inputs = []
    lstm_outputs = []

    lstm_x_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_X_reshape')
    container.add_node('Reshape', operator.inputs[0].full_name, lstm_x_reshape_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1, 1, input_size])
    lstm_inputs.append(lstm_x_reshape_name)

    # Handle LSTM's weight matrices
    matrices_w_forward = np.concatenate([lstm_weights[0].inputGateWeightMatrix.floatValue,
                                         lstm_weights[0].outputGateWeightMatrix.floatValue,
                                         lstm_weights[0].forgetGateWeightMatrix.floatValue,
                                         lstm_weights[0].blockInputWeightMatrix.floatValue])
    matrices_w_backward = np.concatenate([lstm_weights[1].inputGateWeightMatrix.floatValue,
                                          lstm_weights[1].outputGateWeightMatrix.floatValue,
                                          lstm_weights[1].forgetGateWeightMatrix.floatValue,
                                          lstm_weights[1].blockInputWeightMatrix.floatValue])
    matrices_w_name = scope.get_unique_variable_name(lstm_op_name + '_W')
    container.add_initializer(matrices_w_name, onnx_proto.TensorProto.FLOAT, [2, 4 * hidden_size, input_size],
                              np.concatenate([matrices_w_forward, matrices_w_backward]))
    lstm_inputs.append(matrices_w_name)

    # Handle LSTM's recursion matrices
    matrices_r_forward = np.concatenate([lstm_weights[0].inputGateRecursionMatrix.floatValue,
                                         lstm_weights[0].outputGateRecursionMatrix.floatValue,
                                         lstm_weights[0].forgetGateRecursionMatrix.floatValue,
                                         lstm_weights[0].blockInputRecursionMatrix.floatValue])
    matrices_r_backward = np.concatenate([lstm_weights[1].inputGateRecursionMatrix.floatValue,
                                          lstm_weights[1].outputGateRecursionMatrix.floatValue,
                                          lstm_weights[1].forgetGateRecursionMatrix.floatValue,
                                          lstm_weights[1].blockInputRecursionMatrix.floatValue])
    matrices_r_name = scope.get_unique_variable_name(lstm_op_name + '_R')
    container.add_initializer(matrices_r_name, onnx_proto.TensorProto.FLOAT, [2, 4 * hidden_size, hidden_size],
                              np.concatenate([matrices_r_forward, matrices_r_backward]))
    lstm_inputs.append(matrices_r_name)

    # Handle bias vectors
    vectors_b = np.zeros(shape=(2, 8, hidden_size))
    if lstm_params.hasBiasVectors:
        vectors_b[0, 0, :] = lstm_weights[0].inputGateBiasVector.floatValue
        vectors_b[0, 1, :] = lstm_weights[0].outputGateBiasVector.floatValue
        vectors_b[0, 2, :] = lstm_weights[0].forgetGateBiasVector.floatValue
        vectors_b[0, 3, :] = lstm_weights[0].blockInputBiasVector.floatValue
        vectors_b[1, 0, :] = lstm_weights[1].inputGateBiasVector.floatValue
        vectors_b[1, 1, :] = lstm_weights[1].outputGateBiasVector.floatValue
        vectors_b[1, 2, :] = lstm_weights[1].forgetGateBiasVector.floatValue
        vectors_b[1, 3, :] = lstm_weights[1].blockInputBiasVector.floatValue
    if lstm_params.forgetBias:
        # One may think we should do something like b[0, 2, :] += 1. and b[1, 2, :] += 1.,
        # but it's not correct as CoreML has added 1 into those bias vectors.
        pass
    if lstm_params.hasBiasVectors or lstm_params.forgetBias:
        vectors_b_name = scope.get_unique_variable_name(lstm_op_name + '_B')
        container.add_initializer(vectors_b_name, onnx_proto.TensorProto.FLOAT,
                                  [2, 8 * hidden_size], vectors_b.flatten())
        lstm_inputs.append(vectors_b_name)
    else:
        lstm_inputs.append('')

    # Due to the position sensitivity in ONNX argument parsing, we add an empty string for the non-existing
    # sequence length
    lstm_inputs.append('')

    # Handle initial hidden state if necessary
    if len(operator.inputs) > 1:
        lstm_h_init_name = scope.get_unique_variable_name(lstm_op_name + '_h_init')
        container.add_node('Concat', [operator.inputs[1].full_name, operator.inputs[3].full_name],
                           lstm_h_init_name, name=scope.get_unique_operator_name('Concat'), axis=0)

        lstm_h_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_h_init_reshape')
        container.add_node('Reshape', lstm_h_init_name, lstm_h_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[2, 1, hidden_size])

        # Add zero initializers to forward and backward initial hidden states so that they become optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
        container.add_initializer(operator.inputs[3].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[3].type.shape,
                                  np.zeros(shape=operator.inputs[3].type.shape).flatten())
        lstm_inputs.append(lstm_h_init_reshape_name)
    else:
        lstm_inputs.append('')

    # Handle initial cell state if needed
    if len(operator.inputs) > 2:
        lstm_c_init_name = scope.get_unique_variable_name(lstm_op_name + '_c_init')
        container.add_node('Concat', [operator.inputs[2].full_name, operator.inputs[4].full_name],
                           lstm_c_init_name, name=scope.get_unique_operator_name('Concat'), axis=0)

        lstm_c_init_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_c_init_reshape')
        container.add_node('Reshape', lstm_c_init_name, lstm_c_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[2, 1, hidden_size])

        lstm_inputs.append(lstm_c_init_reshape_name)
        # Add zero initializers to forward and backward initial cell states so that they become optional
        container.add_initializer(operator.inputs[2].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[2].type.shape,
                                  np.zeros(shape=operator.inputs[2].type.shape).flatten())
        container.add_initializer(operator.inputs[4].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[4].type.shape,
                                  np.zeros(shape=operator.inputs[4].type.shape).flatten())
    else:
        lstm_inputs.append('')

    # Handle peephole vectors if necessary
    if lstm_params.hasPeepholeVectors:
        p_forward = np.concatenate([lstm_weights[0].inputGatePeepholeVector.floatValue,
                                    lstm_weights[0].outputGatePeepholeVector.floatValue,
                                    lstm_weights[0].forgetGatePeepholeVector.floatValue])
        p_backward = np.concatenate([lstm_weights[1].inputGatePeepholeVector.floatValue,
                                     lstm_weights[1].outputGatePeepholeVector.floatValue,
                                     lstm_weights[1].forgetGatePeepholeVector.floatValue])
        p_name = scope.get_unique_variable_name(lstm_op_name + '_P')
        container.add_initializer(p_name, onnx_proto.TensorProto.FLOAT,
                                  [2, 3 * hidden_size], np.concatenate([p_forward, p_backward]))
        lstm_inputs.append(p_name)
    else:
        lstm_inputs.append('')

    # Parse activation functions and add them into ONNX LSTM's attribute dictionary
    activation_types = []
    alphas = []
    betas = []
    for activation in params.activationsForwardLSTM:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    for activation in params.activationsBackwardLSTM:
        activation_type, alpha, beta = extract_rnn_activation_info(activation)
        activation_types.append(activation_type.encode('ascii'))
        if alpha is not None:
            alphas.append(alpha)
        if beta is not None:
            betas.append(beta)
    lstm_attrs['activations'] = activation_types
    if alphas:
        lstm_attrs['activation_alpha'] = alphas
    if betas:
        lstm_attrs['activation_beta'] = betas

    # Add more attributes
    lstm_attrs['direction'] = 'bidirectional'
    lstm_attrs['output_sequence'] = lstm_params.sequenceOutput
    lstm_attrs['hidden_size'] = hidden_size
    lstm_attrs['clip'] = lstm_params.cellClipThreshold
    lstm_attrs['input_forget'] = lstm_params.coupledInputAndForgetGate

    # Create output part of CoreML LSTM
    if lstm_params.sequenceOutput:
        lstm_y_name = scope.get_unique_variable_name(lstm_op_name + '_Y')
        lstm_outputs.append(lstm_y_name)

        container.add_node('Reshape', lstm_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, 2 * hidden_size])

        if len(operator.outputs) > 1:
            lstm_y_h_name = scope.get_unique_variable_name(lstm_op_name + '_Y_h')
            lstm_outputs.append(lstm_y_h_name)

            lstm_y_h_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_Y_h_reshape')
            container.add_node('Reshape', lstm_y_h_name, lstm_y_h_reshape_name,
                               name=scope.get_unique_operator_name('Reshape'), shape=[2, hidden_size])

            container.add_node('Split', lstm_y_h_reshape_name,
                               [operator.outputs[1].full_name, operator.outputs[3].full_name],
                               name=scope.get_unique_operator_name('Split'), split=[1, 1, ], axis=0)
    else:
        # Here we ingore ONNX RNN's first output because it's useless.
        lstm_outputs.append(scope.get_unique_variable_name('isolated'))

        # Handle the second output of ONNX LSTM. It will become the first and the second outputs of
        # CoreML's LSTM.
        lstm_y_name = scope.get_unique_variable_name(lstm_op_name + '_Y')
        lstm_outputs.append(lstm_y_name)

        # Directly reshape ONNX LSTM's 2nd output to CoreML LSTM's 1st output.
        container.add_node('Reshape', lstm_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, 2 * hidden_size])

        if len(operator.outputs) > 1:
            lstm_y_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_Y_reshape')

            container.add_node('Reshape', lstm_y_name, lstm_y_reshape_name,
                               name=scope.get_unique_operator_name('Reshape'), shape=[2, hidden_size])

            container.add_node('Split', lstm_y_reshape_name,
                               [operator.outputs[1].full_name, operator.outputs[3].full_name],
                               name=scope.get_unique_operator_name('Split'), split=[1, 1], axis=0)

    # Output cell state if necessary
    if len(operator.outputs) > 2:
        lstm_y_c_name = scope.get_unique_variable_name(lstm_op_name + '_Y_c')
        lstm_outputs.append(lstm_y_c_name)

        lstm_y_c_reshape_name = scope.get_unique_variable_name(lstm_op_name + '_Y_c_reshape')
        container.add_node('Reshape', lstm_y_c_name, lstm_y_c_reshape_name,
                          name=scope.get_unique_operator_name('Reshape'), shape=[2, hidden_size])

        container.add_node('Split', lstm_y_c_reshape_name,
                           [operator.outputs[2].full_name, operator.outputs[4].full_name],
                           name=scope.get_unique_operator_name('Split'), split=[1, 1], axis=0)

    # Create the major LSTM operator
    container.add_node('LSTM', lstm_inputs, lstm_outputs, **lstm_attrs)


def convert_simple_rnn(scope, operator, container):
    # The RNN inputs are feature vector, X, and initial hidden state, h_init. Let C_in and C_out denote
    # the input feature length and the output dimension, respectively. Assume that S is the sequence length
    # (i.e., S-axis is the time axis of a sequence). In CorML, the shapes of X and h_init are [S, C_in] and
    # [1, C_out] respectively. In ONNX, the two shapes become [S, N, C_in] (X) and [D, N, C_out]
    # (h_init), where N is the batch size. To simulate CoreML RNN under ONNX, we need to introduce some extra
    # operators. Note that N=1 and D=1 always hold in ONNX if we are considering RNN from CoreML because there
    # is no batch size in CoreML and CoreML RNN is always uni-directional.
    # 
    # Below we provide a visualization of our conversion for CoreML RNN.
    #
    # Symbols:
    #  
    #  X: input features of CoreML RNN
    #  h_init: initial RNN state of CoreML
    #  Y: CoreML RNN's output. It can be [S, C_out] (if sequence_output is on ) or [1, C_out] (if sequence_output is off)
    #  Y_h: CoreML RNN's last hidden state
    #
    #  X': input features of ONNX RNN
    #  h_init': initial RNN state of ONNX
    #  Y': ONNX RNN's output
    #  Y_h': ONNX RNN's last hidden state
    #
    # Computational graph of CoreML RNN (sequence_output is on):
    #
    # X [S, C_in] ---> CoreML RNN ---> Y [S, C_out]
    #                    ^     |
    #                    |     |
    # h_init [1, C_out] -'     '---> Y_h [1, C_out]
    #
    # Computational graph we use for represent CoreML RNN into ONNX (sequence_output is on):
    #
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] -----> ONNX RNN --> Y' [S, 1, C_out] --> Reshape --> Y [S, C_out]
    #                                                        ^ |
    #                                                        | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C_out]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y_h [1, C_out]
    #
    # Computational graph of CoreML RNN (sequence_output is off):
    #
    # X [S, C_in] ---> CoreML RNN ---> Y [1, C_out]
    #                    ^     |
    #                    |     |
    # h_init [1, C_cou] -'     '---> Y_h [1, C_out] Note that in this case, Y=Y_h.
    #
    # Computational graph we use to represent CoreML RNN into ONNX (sequence_output is off):
    #
    # X [S, C_in] --> Reshape --> X' [S, 1, C_in] -----> ONNX RNN --> Y' [S, 1, C_out] Here Y' is useless.
    #                                                        ^ |
    #                                                        | |
    # h_init [1, C_out] --> Reshape -> h_init' [1, 1, C_out]-' '--> Y_h' [1, 1, C_out] ---> Reshape --> Y [1, C_out]
    #                                                                                                   |
    #                                                                                                   v
    #                                                                                                 Identity
    #                                                                                                   |
    #                                                                                                   v
    #                                                                                               Y_h [1, C_out]

    params = operator.raw_operator.simpleRecurrent
    input_size = params.inputVectorSize
    hidden_size = params.outputVectorSize

    X_name = operator.inputs[0].full_name
    X_reshape_name = scope.get_unique_variable_name('X')
    container.add_node('Reshape', X_name, X_reshape_name, name=scope.get_unique_operator_name('Reshape'),
                       shape=[-1, 1, input_size])

    rnn_op_name = scope.get_unique_operator_name('RNN')
    rnn_attrs = {'name': rnn_op_name}
    rnn_inputs = [X_reshape_name]

    # Load RNN's weight matrix and add it into RNN's input list
    rnn_w_name = scope.get_unique_variable_name(rnn_op_name + '_W')
    container.add_initializer(rnn_w_name, onnx_proto.TensorProto.FLOAT,
                              [1, hidden_size, input_size], params.weightMatrix.floatValue)
    rnn_inputs.append(rnn_w_name)

    # Load RNN's recursion matrix and add it into RNN's input list
    rnn_r_name = scope.get_unique_variable_name(rnn_op_name + '_R')
    container.add_initializer(rnn_r_name, onnx_proto.TensorProto.FLOAT,
                              [1, hidden_size, hidden_size], params.recursionMatrix.floatValue)
    rnn_inputs.append(rnn_r_name)

    if params.hasBiasVector:
        # Load RNN's bias vector and add it into RNN's input list
        rnn_b_name = scope.get_unique_variable_name(rnn_op_name + '_B')
        rnn_b_content = np.concatenate([params.biasVector.floatValue, np.zeros(hidden_size)]).flatten()
        container.add_initializer(rnn_b_name, onnx_proto.TensorProto.FLOAT, [1, 2 * hidden_size], rnn_b_content)
        rnn_inputs.append(rnn_b_name)
    else:
        # Input names are position-sensitive, so for optional but missing inputs, we need to provide an empty string.
        rnn_inputs.append('')

    # The input, sequence_lens, in ONNX is alwasy optional for this conversion, so here is always an empty string.
    rnn_inputs.append('')

    # If initial hidden state is provided, we add it into RNN's input list after adjusting its shape.
    if len(operator.inputs) == 2:
        rnn_h_init_reshape_name = scope.get_unique_variable_name(rnn_op_name + '_h_init')
        container.add_node('Reshape', operator.inputs[1].full_name, rnn_h_init_reshape_name,
                           name=scope.get_unique_operator_name('Reshape'),
                           shape=[1, 1, hidden_size])
        rnn_inputs.append(rnn_h_init_reshape_name)
        # Add a zero initializer to initial hidden state so that this variable becomes optional
        container.add_initializer(operator.inputs[1].full_name, onnx_proto.TensorProto.FLOAT,
                                  operator.inputs[1].type.shape,
                                  np.zeros(shape=operator.inputs[1].type.shape).flatten())
    else:
        # Input names are position-sensitive, so for optional but missing inputs, we need to provide an empty string.
        rnn_inputs.append('')

    # Add RNN's information of activation function
    activation, alpha, beta = extract_rnn_activation_info(params.activation)
    rnn_attrs['activations'] = [activation.encode('ascii')]
    if alpha is not None:
        rnn_attrs['activation_alpha'] = [alpha]
    if beta is not None:
        rnn_attrs['activation_beta'] = [beta]

    rnn_attrs['direction'] = 'reverse' if params.reverseInput else 'forward'
    rnn_attrs['output_sequence'] = params.sequenceOutput
    rnn_attrs['hidden_size'] = hidden_size

    # Set up outputs' of RNN
    rnn_outputs = []
    if params.sequenceOutput:
        # Create ONNX's RNN output, which needs to be reshaped to fit CoreML standard.
        rnn_y_name = scope.get_unique_variable_name(rnn_op_name + '_Y')
        rnn_outputs.append(rnn_y_name)

        # Connect ONNX's output and CoreML's output via a reshape operator
        container.add_node('Reshape', rnn_y_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[-1, hidden_size])

        # Handel the second RNN output (aka last hidden state), which is optional.
        if len(operator.outputs) == 2:
            # Create ONNX's RNN output, which needs to be reshaped to fit CoreML standard.
            rnn_h_name = scope.get_unique_variable_name(rnn_op_name + '_Y_h')
            rnn_outputs.append(rnn_h_name)

            # Connect ONNX's output and CoreML's output via a reshape operator
            container.add_node('Reshape', rnn_h_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Reshape'),
                               shape=[1, hidden_size])
    else:
        # Here we ignore ONNX RNN's first output by assigning it an isolated name. Isolated names
        # are not connected with anything else.
        rnn_outputs.append(scope.get_unique_variable_name('isolated'))

        # According to CoreML, the two outputs are always identical, so we just need to compute one of
        # them and produce the other one using an identiy operator.
        rnn_h_name = scope.get_unique_variable_name(rnn_op_name + '_Y_h')
        rnn_outputs.append(rnn_h_name)

        # Reshape last hidden state's ONNX format to its CoreML format
        container.add_node('Reshape', rnn_h_name, operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Reshape'), shape=[1, hidden_size])

        if len(operator.outputs) == 2:
            # Copy the first output to the second output
            container.add_node('Identity', operator.outputs[0].full_name, operator.outputs[1].full_name,
                               name=scope.get_unique_operator_name('Identity'))

    # Finally, we use the collected information to build ONNX's RNN
    container.add_node('RNN', rnn_inputs, rnn_outputs, **rnn_attrs)


def convert_tensor_to_probability_map(scope, operator, container):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Too many input or output variables')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise TypeError('Only float tensor is supported')

    attrs = {'name': scope.get_unique_operator_name('ZipMap')}

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        model = operator.raw_operator.neuralNetworkClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = [s.encode('ascii') for s in model.stringClassLabels.vector]
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = model.int64ClassLabels.vector
        else:
            raise ValueError('Unknown label type found')
    elif model_type == 'pipelineClassifier':
        model = operator.raw_operator.pipelineClassifier
        if model.WhichOneof('ClassLabels') == 'stringClassLabels':
            attrs['classlabels_strings'] = [s.encode('ascii') for s in model.stringClassLabels.vector]
        elif model.WhichOneof('ClassLabels') == 'int64ClassLabels':
            attrs['classlabels_int64s'] = model.int64ClassLabels.vector
        else:
            raise ValueError('Unknown label type found')
    else:
        raise TypeError('Only neural network classifiers and pipeline classifiers are supported')

    container.add_node('ZipMap', [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)


def convert_dictionary_vectorizer(scope, operator, container):
    op_type = 'DictVectorizer'
    attrs = {'name': operator.full_name}
    raw_model = operator.raw_operator.dictVectorizer
    if raw_model.HasField('stringToIndex'):
        attrs['string_vocabulary'] = raw_model.stringToIndex.vector
    else:
        attrs['int64_vocabulary'] = raw_model.int64ToIndex.vector

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)


def convert_feature_vectorizer(scope, operator, container):
    op_type = 'FeatureVectorizer'
    attrs = {'name': operator.full_name}

    inputs = []
    input_dims = []
    for variable in operator.inputs:
        if type(variable.type) in [Int64TensorType, Int64Type]:
            # We use scaler to convert integers into floats because output is a single tensor and all tensor elements
            # should be in the same type.
            scaler_name = scope.get_unique_operator_name('Scaler')
            scaled_name = scope.get_unique_variable_name(variable.full_name + '_scaled')
            scaler_attrs = {'name': scaler_name, 'scale': [1.], 'offset': [0.]}
            container.add_node('Scaler', [variable.full_name], [scaled_name], **scaler_attrs)
            inputs.append(scaled_name)
        else:
            inputs.append(variable.full_name)
        # We assume feature vectorizer always combines inputs with shapes [1, C] or [C]
        input_dims.append(variable.type.shape[1])
    attrs['inputdimensions'] = input_dims

    container.add_node(op_type, inputs, [operator.outputs[0].full_name], **attrs)


def convert_tree_ensemble_model(scope, operator, container):
    COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE = {
        0: 'BRANCH_LEQ',
        1: 'BRANCH_LT',
        2: 'BRANCH_GTE',
        3: 'BRANCH_GT',
        4: 'BRANCH_EQ',
        5: 'BRANCH_NEQ',
        6: 'LEAF'
    }

    COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM = {
        0: 'NONE',
        1: 'SOFTMAX',
        2: 'LOGISTIC',
        3: 'SOFTMAX_ZERO'
    }

    def get_onnx_tree_mode(cm_tree_behavior):
        if cm_tree_behavior in COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE:
            return COREML_TREE_NODE_BEHAVIOR_TO_ONNX_TREE_NODE_MODE[cm_tree_behavior]
        raise RuntimeError('CoreML tree node behavior not supported {0}'.format(cm_tree_behavior))

    def get_onnx_tree_post_transform(cm_tree_post_transform):
        if cm_tree_post_transform in COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM:
            return COREML_TREE_POST_TRANSFORM_TO_ONNX_TREE_POST_TRANSFORM[cm_tree_post_transform]
        raise RuntimeError('CoreML tree post transform not supported {0}'.format(cm_tree_post_transform))

    raw_model = operator.raw_operator
    attrs = {'name': operator.full_name}
    if raw_model.WhichOneof('Type') == 'treeEnsembleClassifier':
        op_type = 'TreeEnsembleClassifier'
        prefix = 'class'
        nodes = raw_model.treeEnsembleClassifier.treeEnsemble.nodes
        attrs['base_values'] = raw_model.treeEnsembleClassifier.treeEnsemble.basePredictionValue
        attrs['post_transform'] = get_onnx_tree_post_transform(raw_model.treeEnsembleClassifier.postEvaluationTransform)
        zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
        # [TODO] We should use WhichOnelf('ClassLabels') to replace HasField below for proto3 capability
        if raw_model.treeEnsembleClassifier.HasField('int64ClassLabels'):
            class_labels = raw_model.treeEnsembleClassifier.int64ClassLabels.vector
            attrs['classlabels_int64s'] = class_labels
            zipmap_attrs['classlabels_int64s'] = class_labels
        else:
            class_labels = [s.encode('ascii') for s in raw_model.treeEnsembleClassifier.stringClassLabels.vector]
            attrs['classlabels_strings'] = class_labels
            zipmap_attrs['classlabels_strings'] = class_labels
    elif raw_model.WhichOneof('Type') == 'treeEnsembleRegressor':
        op_type = 'TreeEnsembleRegressor'
        prefix = 'target'
        nodes = raw_model.treeEnsembleRegressor.treeEnsemble.nodes
        attrs['base_values'] = raw_model.treeEnsembleRegressor.treeEnsemble.basePredictionValue
        attrs['post_transform'] = get_onnx_tree_post_transform(raw_model.treeEnsembleRegressor.postEvaluationTransform)
    else:
        raise ValueError('Unknown tree model type')

    leaf_treeids = [node.treeId for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]
    leaf_nodeids = [node.nodeId for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]
    leaf_ids = [weight.evaluationIndex for node in nodes if 6 == node.nodeBehavior for weight in node.evaluationInfo]

    leaf_weights = [weight.evaluationValue for node in nodes if 6 == node.nodeBehavior for weight in
                    node.evaluationInfo]

    assert (len(leaf_ids) == len(leaf_weights))
    assert (len(leaf_weights) == len(leaf_nodeids))
    assert (len(leaf_nodeids) == len(leaf_treeids))

    nodes_nodeids = [x.nodeId for x in nodes]
    nodes_treeids = [x.treeId for x in nodes]
    nodes_featureids = [x.branchFeatureIndex for x in nodes]
    nodes_values = [x.branchFeatureValue for x in nodes]
    nodes_truenodeids = [x.trueChildNodeId for x in nodes]
    nodes_falsenodeids = [x.falseChildNodeId for x in nodes]
    nodes_missing_value_tracks_true = [x.missingValueTracksTrueChild for x in nodes]
    nodes_hitrates = [x.relativeHitRate for x in nodes]
    nodes_modes = [get_onnx_tree_mode(x.nodeBehavior) for x in nodes]

    attrs['nodes_treeids'] = nodes_treeids
    attrs['nodes_nodeids'] = nodes_nodeids
    attrs['nodes_featureids'] = nodes_featureids
    attrs['nodes_values'] = nodes_values
    attrs['nodes_hitrates'] = nodes_hitrates
    attrs['nodes_modes'] = nodes_modes
    attrs['nodes_truenodeids'] = nodes_truenodeids
    attrs['nodes_falsenodeids'] = nodes_falsenodeids
    attrs['nodes_missing_value_tracks_true'] = nodes_missing_value_tracks_true
    attrs[prefix + '_treeids'] = leaf_treeids
    attrs[prefix + '_nodeids'] = leaf_nodeids
    attrs[prefix + '_ids'] = leaf_ids
    attrs[prefix + '_weights'] = leaf_weights

    # For regression, we can simply construct a model. For classifier, due to the different representation of
    # classes' probabilities, we need to add some operators for type conversion.
    if raw_model.WhichOneof('Type') == 'treeEnsembleRegressor':
        # Create ONNX representation of this operator. If there is only one input, its full topology is
        #
        # input features ---> TreeEnsembleRegressor ---> output
        #
        # If there are multiple (e.g., "N" features) input features, we need to concatenate them all together before feeding them into
        # ONNX tree-based model. It leads to the following computational graph.
        #
        # input feature 1 -----.
        #        ...           |
        #        ...           v
        #        ...      ---> Feature Vectorizer ---> TreeEnsembleRegressor ---> output
        #        ...           ^
        #        ...           |
        # input feature N -----'
        if len(operator.inputs) > 1:
            feature_vector_name = scope.get_unique_variable_name('feature_vector')
            container.add_node('FeatureVectorizer', operator.input_full_names, feature_vector_name,
                               name=scope.get_unique_operator_name('FeatureVectorizer'),
                               inputdimensions=[variable.type.shape[1] for variable in operator.inputs])
            container.add_node(op_type, feature_vector_name, operator.output_full_names, **attrs)
        else:
            container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)
    else:
        # For classifiers, due to the different representation of classes' probabilities, we need to add some
        # operators for type conversion. It turns out that we have the following topology.
        # input features ---> TreeEnsembleClassifier ---> label (must present)
        #                               |
        #                               '--> probability tensor ---> ZipMap ---> probability map (optional)
        #
        # Similar to the regressor's case, if there are multiple input features, we need to concatenate them all
        # together before feeding them into ONNX tree-based model. It leads to the following computational graph.
        #
        # input feature 1 -----.
        #        ...           |
        #        ...           v
        #        ...      ---> Feature Vectorizer ---> TreeEnsembleClassifier ---> label (must present)
        #        ...           ^                                 |
        #        ...           |                                 '--> probability tensor ---> ZipMap ---> probability
        # input feature N -----'                                                                          map (optional)

        # Set up input feature(s)
        if len(operator.inputs) > 1:
            feature_vector_name = scope.get_unique_variable_name('feature_vector')
            container.add_node('FeatureVectorizer', operator.input_full_names, feature_vector_name,
                               name=scope.get_unique_operator_name('FeatureVectorizer'),
                               inputdimensions=[variable.type.shape[1] for variable in operator.inputs])
        else:
            feature_vector_name = operator.inputs[0].full_name

        # Find label name and probability name
        proba_output_name = None
        for variable in operator.outputs:
            if raw_model.description.predictedFeatureName == variable.raw_name:
                label_output_name = variable.full_name
            if raw_model.description.predictedProbabilitiesName != '' and raw_model.description.predictedProbabilitiesName == variable.raw_name:
                proba_output_name = variable.full_name

        inputs = [variable.full_name for variable in operator.inputs]
        proba_tensor_name = scope.get_unique_variable_name('ProbabilityTensor')

        if proba_output_name is not None:
            # Add tree model ONNX node with probability output
            container.add_node(op_type, feature_vector_name, [label_output_name, proba_tensor_name], **attrs)

            # Add ZipMap to convert probability tensor into probability map
            container.add_node('ZipMap', [proba_tensor_name], [proba_output_name], **zipmap_attrs)
        else:
            # Add support vector classifier without probability output
            container.add_node(op_type, feature_vector_name, [label_output_name, proba_tensor_name], **attrs)


def convert_glm_classifier(scope, operator, container):
    from coremltools.proto.GLMClassifier_pb2 import GLMClassifier
    op_type = 'LinearClassifier'
    attrs = {'name': operator.full_name}
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    glm = operator.raw_operator.glmClassifier

    transform_table = {GLMClassifier.Logit: 'LOGISTIC', GLMClassifier.Probit: 'PROBIT'}
    if glm.postEvaluationTransform not in transform_table:
        raise ValueError('Unsupported post-transformation: {}'.format(glm.postEvaluationTransform))
    attrs['post_transform'] = transform_table[glm.postEvaluationTransform]

    encoding_table = {GLMClassifier.ReferenceClass: True, GLMClassifier.OneVsRest: False}
    if glm.classEncoding not in encoding_table:
        raise ValueError('Unsupported class encoding: {}'.format(glm.classEncoding))
    attrs['multi_class'] = encoding_table[glm.classEncoding]

    # Determine the dimensionality of the model weights.
    dim_target = len(glm.weights)
    dim_feature = len(glm.weights[0].value)

    matrix_w = np.ndarray(shape=(dim_target, dim_feature))
    for i, w in enumerate(glm.weights):
        matrix_w[i, :] = w.value

    if glm.WhichOneof('ClassLabels') == 'stringClassLabels':
        class_labels = [s.encode('ascii') for s in glm.stringClassLabels.vector]
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    elif glm.WhichOneof('ClassLabels') == 'int64ClassLabels':
        class_labels = glm.int64ClassLabels.vector
        attrs['classlabels_ints'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    else:
        raise ValueError('Unknown class label type')

    coefficients = matrix_w.flatten().tolist()
    intercepts = cast_list(float, glm.offset)
    if len(class_labels) == 2:
        # Handle the binary case for coefficients and intercepts
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    attrs['coefficients'] = coefficients
    attrs['intercepts'] = intercepts

    # For classifiers, due to the different representation of classes' probabilities, we need to add some
    # operators for type conversion. It turns out that we have the following topology.
    # input features ---> GLMClassifier ---> label (must present)
    #                           |
    #                           '--> probability tensor ---> Normalizer ---> normalized ---> ZipMap ---> probability map
    #                                                (depending on whether probability output exists in CoreML model,
    #                                                 variables/operators after probability tensor may disappear)
    raw_model = operator.raw_operator
    # Find label name and probability name
    proba_output_name = None
    for variable in operator.outputs:
        if raw_model.description.predictedFeatureName == variable.raw_name:
            label_output_name = variable.full_name
        if raw_model.description.predictedProbabilitiesName != '' and raw_model.description.predictedProbabilitiesName == variable.raw_name:
            proba_output_name = variable.full_name

    inputs = [variable.full_name for variable in operator.inputs]
    proba_tensor_name = scope.get_unique_variable_name('ProbabilityTensor')

    if proba_output_name is not None:
        # Add tree model ONNX node with probability output
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], **attrs)

        # Add a normalizer to make sure that the sum of all classes' probabilities is 1. It doesn't affect binary
        # classification. For multi-class clssifiers, if one applies sigmoid function independently to all raw scores,
        # we have to add a normalization so that the sum of all probabilities remains 1. Of course, if softmax is used
        # to convert raw scores into probabilities, this normalization doesn't change anything.
        if len(class_labels) > 2:
            normalized_proba_tensor_name = scope.get_unique_variable_name(proba_tensor_name + '_normalized')
            container.add_node('Normalizer', proba_tensor_name, normalized_proba_tensor_name,
                               name=scope.get_unique_operator_name('Normalizer'), norm='L1')
        else:
            # If we don't need a normalization, we just pass the original probability tensor to the following ZipMap
            normalized_proba_tensor_name = proba_tensor_name

        # Add ZipMap to convert normalized probability tensor into probability map
        container.add_node('ZipMap', [normalized_proba_tensor_name], [proba_output_name], **zipmap_attrs)
    else:
        # Add linear classifier with isolated probability output, which means that the probability
        # tensor won't be accessed by any others.
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], **attrs)


def convert_glm_regressor(scope, operator, container):
    from coremltools.proto.GLMRegressor_pb2 import GLMRegressor

    op_type = 'LinearRegressor'
    glm = operator.raw_operator.glmRegressor
    attrs = {'name': operator.full_name}

    transform_table = {GLMRegressor.NoTransform: 'NONE', GLMRegressor.Logit: 'LOGISTIC', GLMRegressor.Probit: 'PROBIT'}
    if glm.postEvaluationTransform in transform_table:
        attrs['post_transform'] = transform_table[glm.postEvaluationTransform]
    else:
        raise ValueError('Unsupported post-transformation: {}'.format(glm.postEvaluationTransform))

    # Determine the dimensionality of the model weights. Conceptually,
    # the shape of the weight matrix in CoreML is E-by-F, where E and F
    # respectively denote the number of target variables and the number
    # of used features. Note that in ONNX, the shape is F-by-E.
    dim_target = len(glm.weights)
    dim_feature = len(glm.weights[0].value)

    matrix_w = np.ndarray(shape=(dim_feature, dim_target))
    for i, w in enumerate(glm.weights):
        matrix_w[:, i] = w.value

    attrs['targets'] = dim_target
    attrs['coefficients'] = matrix_w.flatten()
    attrs['intercepts'] = glm.offset

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_array_feature_extractor(scope, operator, container):
    op_type = 'ArrayFeatureExtractor'
    attrs = {'name': operator.full_name}

    target_indexes = operator.raw_operator.arrayFeatureExtractor.extractIndex
    index_buffer_name = scope.get_unique_variable_name('target_indexes')
    container.add_initializer(index_buffer_name, onnx_proto.TensorProto.INT64, [len(target_indexes)], target_indexes)

    inputs = [operator.inputs[0].full_name, index_buffer_name]
    outputs = [operator.outputs[0].full_name]

    container.add_node(op_type, inputs, outputs, **attrs)


def convert_one_hot_encoder(scope, operator, container):
    op_type = 'OneHotEncoder'
    attrs = {'name': operator.full_name}

    raw_model = operator.raw_operator.oneHotEncoder
    if raw_model.HasField('int64Categories'):
        attrs['cats_int64s'] = raw_model.int64Categories.vector
    if raw_model.HasField('stringCategories'):
        attrs['cats_strings'] = raw_model.stringCategories.vector

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)


def convert_normalizer(scope, operator, container):
    op_type = 'Normalizer'
    attrs = {'name': operator.full_name}
    norms = ['MAX', 'L1', 'L2']
    norm_type = operator.raw_operator.normalizer.normType
    if norm_type in range(3):
        attrs['norm'] = norms[norm_type]
    else:
        raise RuntimeError('Invalid norm type: ' + norm_type)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_imputer(scope, operator, container):
    op_type = 'Imputer'
    attrs = {'name': operator.full_name}
    imputer = operator.raw_operator.imputer
    if imputer.HasField('replaceDoubleValue'):
        attrs['replaced_value_float'] = imputer.replaceDoubleValue
    elif imputer.HasField('replaceInt64Value'):
        attrs['replaced_value_int64'] = imputer.replaceInt64Value
    if imputer.HasField('imputedDoubleArray'):
        attrs['imputed_value_floats'] = imputer.imputedDoubleArray.vector
    elif imputer.HasField('imputedInt64Array'):
        attrs['imputed_value_int64s'] = imputer.imputedInt64Array.vector
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_scaler(scope, operator, container):
    op_type = 'Scaler'
    attrs = {'name': operator.full_name}
    scaler = operator.raw_operator.scaler

    scale = [x for x in scaler.scaleValue]
    offset = [-x for x in scaler.shiftValue]

    attrs['scale'] = scale
    attrs['offset'] = offset

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_padding(scope, operator, container):
    op_type = 'Pad'
    attrs = {'name': operator.full_name}
    params = operator.raw_operator.padding

    pad_table = {'constant': 'constant', 'reflection': 'reflect', 'replication': 'edge'}
    pad_type = params.WhichOneof('PaddingType')
    if pad_type not in pad_table:
        raise ValueError('Unsupported padding mode: {}'.format(pad_type))
    attrs['mode'] = pad_table[pad_type]

    # CoreML only pads for their H- and W- axes. Here we assume input tensor's shape is [N, C, H, W].
    pads = [0, 0, 0, 0, 0, 0, 0, 0]
    if len(params.paddingAmounts.borderAmounts) > 0:
        pads[2] = params.paddingAmounts.borderAmounts[0].startEdgeSize
        pads[3] = params.paddingAmounts.borderAmounts[1].startEdgeSize
        pads[6] = params.paddingAmounts.borderAmounts[0].endEdgeSize
        pads[7] = params.paddingAmounts.borderAmounts[1].endEdgeSize
    attrs['pads'] = pads

    if pad_type == 'constant':
        attrs['values'] = params.constant.value

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_crop(scope, operator, container):
    if len(operator.input_full_names) > 2:
        raise ValueError('Unlike CoreML, ONNX only supports cropping with a single input')

    op_type = 'Crop'
    attrs = {'name': operator.full_name}
    border = operator.raw_operator.crop.cropAmounts.borderAmounts
    left = border[1].startEdgeSize
    top = border[0].startEdgeSize
    right = border[1].endEdgeSize
    bottom = border[0].endEdgeSize

    attrs['border'] = [left, top, right, bottom]

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


def convert_add(scope, operator, container):
    op_type = 'Add'
    attrs = {'name': operator.full_name}

    if len(operator.input_full_names) == 1:
        scaler_name = scope.get_unique_variable_name(op_type + '_B')
        container.add_initializer(scaler_name, onnx_proto.TensorProto.FLOAT, [1], operator.raw_operator.add.alpha)
        inputs = [operator.inputs[0].full_name, scaler_name]
    else:
        inputs = operator.input_full_names

    container.add_node(op_type, inputs, operator.output_full_names, **attrs)


def convert_average(scope, operator, container):
    container.add_node('Mean', operator.input_full_names, operator.output_full_names, name=operator.full_name)


def convert_bias(scope, operator, container):
    params = operator.raw_operator.bias
    attrs = {'name': operator.full_name}

    axis, shape = deduce_broadcast_axis_and_shape(params.shape)
    if axis is not None:
        attrs['axis'] = axis
    attrs['broadcast'] = 1  # True

    bias_tensor_name = scope.get_unique_variable_name(operator.full_name + '_B')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, shape, params.bias.floatValue)

    container.add_node('Add', [operator.inputs[0].full_name, bias_tensor_name], operator.output_full_names, **attrs)


def convert_scale(scope, operator, container):
    params = operator.raw_operator.scale
    op1_type = 'Mul'
    attrs1 = {'name': scope.get_unique_operator_name(op1_type)}
    scale_axis, scale_shape = deduce_broadcast_axis_and_shape(params.shapeScale)
    scale_name = scope.get_unique_variable_name(op1_type + '_B')
    container.add_initializer(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, params.scale.floatValue)

    if scale_axis is not None:
        attrs1['axis'] = scale_axis
    # No matter what shape it is, we activate broadcasting because input shape is larger or equal to the scalar.
    attrs1['broadcast'] = 1

    if not params.hasBias:
        container.add_node(op1_type, [operator.inputs[0].full_name, scale_name], operator.output_full_names, **attrs1)
    else:
        intra_variable_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_scaled')
        container.add_node(op1_type, operator.input_full_names, [intra_variable_name], **attrs1)
        op2_type = 'Add'
        attrs2 = {'name': scope.get_unique_operator_name(op2_type)}
        bias_axis, bias_shape = deduce_broadcast_axis_and_shape(params.shapeBias)
        if bias_axis is not None:
            attrs2['axis'] = scale_axis
        attrs2['broadcast'] = 1
        bias_name = scope.get_unique_variable_name(op2_type + '_B')
        container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT, bias_shape, params.bias.floatValue)
        container.add_node(op2_type, [intra_variable_name, bias_name], operator.output_full_names, **attrs2)


def convert_batch_normalization(scope, operator, container):
    params = operator.raw_operator.batchnorm

    if params.instanceNormalization and not params.computeMeanVar:
        raise ValueError('It is impossible to do instance normalization without re-computing mean and variance')

    if params.instanceNormalization and params.computeMeanVar:
        op_type = 'InstanceNormalization'
    else:
        op_type = 'BatchNormalization'

    attrs = {'name': operator.full_name}
    inputs = [operator.inputs[0].full_name]
    outputs = [operator.outputs[0].full_name]
    scale_tensor_name = scope.get_unique_variable_name(op_type + '_scale')
    container.add_initializer(scale_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                              params.gamma.floatValue)
    inputs.append(scale_tensor_name)
    bias_tensor_name = scope.get_unique_variable_name(op_type + '_B')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels], params.beta.floatValue)
    inputs.append(bias_tensor_name)

    attrs['epsilon'] = params.epsilon
    attrs['spatial'] = 1  # True

    if op_type == 'BatchNormalization':
        mean_tensor_name = scope.get_unique_variable_name(op_type + '_mean')
        container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                                  params.mean.floatValue)
        inputs.append(mean_tensor_name)
        variance_tensor_name = scope.get_unique_variable_name(op_type + '_variance')
        container.add_initializer(variance_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                                  params.variance.floatValue)
        inputs.append(variance_tensor_name)
        attrs['momentum'] = 0.

        if not params.instanceNormalization and params.computeMeanVar:
            # This is training mode, so some variables may be updated. To update "mean"
            # and "var," we put some results back to the associated input tensors. We also
            # allocate two extra output buffers to store some intermediate results.
            outputs += inputs[1:3]
            attrs['is_test'] = 0  # False
        elif not params.instanceNormalization and not params.computeMeanVar:
            attrs['is_test'] = 1  # True
    else:
        attrs['is_test'] = 1  # True

    container.add_node(op_type, inputs, outputs, **attrs)

def convert_identity(scope, operator, container):
    container.add_node('Identity', operator.input_full_names, operator.output_full_names, name=operator.full_name)


def convert_svm_classifier(scope, operator, container):
    params = operator.raw_operator.supportVectorClassifier
    kernel_enum = {'linearKernel': 'LINEAR', 'polyKernel': 'POLY',
                   'rbfKernel': 'RBF', 'sigmoidKernel': 'SIGMOID', 'precomputedKernel': 'PRECOMPUTED'}
    kernel = params.kernel
    kernel_val = kernel.WhichOneof('kernel')
    svc_kernel = kernel_enum[kernel_val]

    if kernel_val == 'rbfKernel':
        svc_kernel_params = [kernel.rbfKernel.gamma, 0.0, 0.0]
    elif kernel_val == 'polyKernel':
        svc_kernel_params = [kernel.polyKernel.gamma,
                             kernel.polyKernel.coef0, kernel.polyKernel.degree]
    elif kernel_val == 'sigmoidKernel':
        svc_kernel_params = [kernel.sigmoidKernel.gamma,
                             kernel.sigmoidKernel.coef0, 0.0]
    elif kernel_val == 'linearKernel':
        svc_kernel_params = [0.0, 0.0, 0.0]

    prob_a = params.probA
    prob_b = params.probB
    # TODO: handle sparse vectors
    svc_vectors_per_class = params.numberOfSupportVectorsPerClass
    vectors = params.denseSupportVectors.vectors
    support_vectors = np.array([v.values for v in vectors])
    support_vectors = support_vectors.flatten()
    svc_support_vectors = support_vectors
    chain_coef = list(itertools.chain.from_iterable([coef.alpha for coef in params.coefficients]))
    svc_coefficients = chain_coef
    svc_rho = [-x for x in params.rho]

    op_type = 'SVMClassifier'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name}
    attrs['kernel_type'] = svc_kernel
    attrs['kernel_params'] = svc_kernel_params
    if prob_a:
        attrs['prob_a'] = prob_a
    if prob_b:
        attrs['prob_b'] = prob_b
    attrs['vectors_per_class'] = svc_vectors_per_class
    attrs['support_vectors'] = svc_support_vectors
    attrs['coefficients'] = svc_coefficients
    attrs['rho'] = svc_rho
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    svc_classes = params.WhichOneof('ClassLabels')
    if svc_classes == 'int64ClassLabels':
        class_labels = params.int64ClassLabels.vector
        attrs['classlabels_ints'] = class_labels
        zipmap_attrs['classlabels_int64s'] = class_labels
    elif svc_classes == 'stringClassLabels':
        class_labels = params.stringClassLabels.vector
        attrs['classlabels_strings'] = class_labels
        zipmap_attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Unknown class label type')

    # For classifiers, due to the different representation of classes' probabilities, we need to add some
    # operators for type conversion. It turns out that we have the following topology.
    # input features ---> SupportVectorClassifier ---> label (must present)
    #                               |
    #                               '--> probability tensor ---> ZipMap ---> probability map (optional)

    raw_model = operator.raw_operator
    # Find label name and probability name
    proba_output_name = None
    for variable in operator.outputs:
        if raw_model.description.predictedFeatureName == variable.raw_name:
            label_output_name = variable.full_name
        if raw_model.description.predictedProbabilitiesName != '' and raw_model.description.predictedProbabilitiesName == variable.raw_name:
            proba_output_name = variable.full_name

    inputs = [variable.full_name for variable in operator.inputs]
    proba_tensor_name = scope.get_unique_variable_name('ProbabilityTensor')

    if proba_output_name is not None:
        # Add support vector classifier in terms of ONNX node with probability output
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], **attrs)

        # Add ZipMap to convert probability tensor into probability map
        container.add_node('ZipMap', [proba_tensor_name], [proba_output_name], **zipmap_attrs)
    else:
        # Add support vector classifier in terms of ONNX node
        container.add_node(op_type, inputs, [label_output_name, proba_tensor_name], **attrs)


def convert_svm_regressor(scope, operator, container):
    params = operator.raw_operator.supportVectorRegressor

    kernel_enum = {'linearKernel': 'LINEAR', 'polyKernel': 'POLY',
                   'rbfKernel': 'RBF', 'sigmoidKernel': 'SIGMOID', 'precomputedKernel': 'PRECOMPUTED'}
    kernel = params.kernel
    kernel_val = kernel.WhichOneof('kernel')
    svr_kernel = kernel_enum[kernel_val]

    if kernel_val == 'rbfKernel':
        svr_kernel_params = [kernel.rbfKernel.gamma, 0.0, 0.0]
    elif kernel_val == 'polyKernel':
        svr_kernel_params = [kernel.polyKernel.gamma,
                             kernel.polyKernel.coef0, kernel.polyKernel.degree]
    elif kernel_val == 'sigmoidKernel':
        svr_kernel_params = [kernel.sigmoidKernel.gamma,
                             kernel.sigmoidKernel.coef0, 0.0]
    elif kernel_val == 'linearKernel':
        svr_kernel_params = [0.0, 0.0, 0.0]

    # TODO: handle sparse vectors
    vectors = params.denseSupportVectors.vectors
    support_vectors = np.array([v.values for v in vectors])
    i = support_vectors.shape[0]
    support_vectors = support_vectors.flatten()
    svr_support_vectors = support_vectors

    svr_coefficients = params.coefficients.alpha
    if isinstance(params.rho, list):
        svr_rho = [-x for x in params.rho]
    else:
        svr_rho = [-params.rho]

    op_type = 'SVMRegressor'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name}
    attrs['kernel_type'] = svr_kernel
    attrs['kernel_params'] = svr_kernel_params
    attrs['support_vectors'] = svr_support_vectors
    attrs['n_supports'] = i
    attrs['coefficients'] = svr_coefficients
    attrs['rho'] = svr_rho

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


converter_table = {'activation': convert_activation,
                   'innerProduct': convert_inner_product,
                   'identity': convert_identity,
                   'softmax': convert_softmax,
                   'convolution': convert_convolution,
                   'pooling': convert_pooling,
                   'scalerPreprocessor': convert_preprocessing_scaler,
                   'flatten': convert_flatten,
                   'permute': convert_permute,
                   'imageToFloatTensor': convert_image_to_float_tensor,
                   'uniDirectionalLSTM': convert_unidirectional_lstm,
                   'embedding': convert_embedding,
                   'concat': convert_concat,
                   'reshape': convert_reshape,
                   'tensorToLabel': convert_tensor_to_label,
                   'tensorToProbabilityMap': convert_tensor_to_probability_map,
                   'gru': convert_gru,
                   'dictVectorizer': convert_dictionary_vectorizer,
                   'featureVectorizer': convert_feature_vectorizer,
                   'treeEnsembleClassifier': convert_tree_ensemble_model,
                   'treeEnsembleRegressor': convert_tree_ensemble_model,
                   'glmClassifier': convert_glm_classifier,
                   'glmRegressor': convert_glm_regressor,
                   'arrayFeatureExtractor': convert_array_feature_extractor,
                   'oneHotEncoder': convert_one_hot_encoder,
                   'imputer': convert_imputer,
                   'scaler': convert_scaler,
                   'normalizer': convert_normalizer,
                   'padding': convert_padding,
                   'batchnorm': convert_batch_normalization,
                   'crop': convert_crop,
                   'add': convert_add,
                   'scale': convert_scale,
                   'average': convert_average,
                   'bias': convert_bias,
                   'dot': convert_dot,
                   'l2normalize': convert_l2_normalization,
                   'loadConstant': convert_load_constant,
                   'lrn': convert_lrn,
                   'max': convert_max,
                   'meanImagePreprocessor': convert_preprocessing_mean_image,
                   'min': convert_min,
                   'mvn': convert_mean_variance_normalization,
                   'multiply': convert_multiply,
                   'reduce': convert_reduce,
                   'reorganizeData': convert_reorganize_data,
                   'sequenceRepeat': convert_sequence_repeat,
                   'slice': convert_slice,
                   'split': convert_split,
                   'unary': convert_unary,
                   'upsample': convert_upsample,
                   'biDirectionalLSTM': convert_bidirectional_lstm,
                   'simpleRecurrent': convert_simple_rnn,
                   'supportVectorClassifier': convert_svm_classifier,
                   'supportVectorRegressor': convert_svm_regressor}
