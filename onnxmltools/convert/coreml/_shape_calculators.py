import copy
import math
from ._data_types import *


def calculate_inner_product_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Inner product layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType:
        raise RuntimeError('Input must be float tensor')

    params = operator.raw_operator.innerProduct
    C_out = params.outputChannels
    C_in = params.inputChannels

    input_shape = input.type.shape
    if len(input_shape) == 4 and (input_shape[2] != 1 or input_shape[3] != 1):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, C, 1, 1]')
    if input_shape[1] != C_in:
        raise RuntimeError('Dimension mismatch along C-axis. Expected %s but got %s' % (C_in, input_shape[1]))

    if len(input_shape) == 4:
        output.type = FloatTensorType([input_shape[0], C_out, 1, 1])
    elif len(input_shape) == 2:
        output.type = FloatTensorType([input_shape[0], C_out])
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')


def calculate_activation_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Activation layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType:
        raise RuntimeError('Input must be float tensor')

    output.type = FloatTensorType([d for d in input.type.shape])  # Similar to identity but only accept floats


def calculate_identity_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Identity layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]
    output.type = copy.deepcopy(input.type)


def calculate_image_to_float_tensor_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Type conversion layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    output.type = FloatTensorType(shape=copy.deepcopy(input.type.shape), color_space=input.type.color_space)


def calculate_softmax_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Softmax layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType:
        raise RuntimeError('Input of softmax layer, %s, must be float tensor' % operator.ful_name)

    output.type = copy.deepcopy(input.type)


def calculate_convolution_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Convolution layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or len(input.type.shape) != 4:
        raise RuntimeError('Input must be 4-D float tensor')

    input_shape = input.type.shape

    params = operator.raw_operator.convolution

    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]

    n_groups = params.nGroups
    kernel_h = kernel_w = 3
    stride_h = stride_w = dilation_h = dilation_w = 1
    if len(params.kernelSize) != 0:
        kernel_h, kernel_w = params.kernelSize
    if len(params.stride) != 0:
        stride_h, stride_w = params.stride
    if len(params.dilationFactor) != 0:
        dilation_h, dilation_w = params.dilationFactor
    kernel_h_dilated = (kernel_h - 1) * dilation_h + 1
    kernel_w_dilated = (kernel_w - 1) * dilation_w + 1
    pad_left = pad_right = pad_bottom = pad_top = 0
    if params.WhichOneof('ConvolutionPaddingType') == 'valid':
        pad_amounts = params.valid.paddingAmounts.borderAmounts
        if len(pad_amounts) != 0:
            pad_top = pad_amounts[0].startEdgeSize
            pad_bottom = pad_amounts[0].endEdgeSize
            pad_left = pad_amounts[1].startEdgeSize
            pad_right = pad_amounts[1].endEdgeSize
        if params.isDeconvolution:
            if isinstance(H, int):
                H_out = (H - 1) * stride_h + kernel_h_dilated - pad_top - pad_bottom
            else:
                H_out = 'None'
            if isinstance(W, int):
                W_out = (W - 1) * stride_w + kernel_w_dilated - pad_right - pad_left
            else:
                W_out = 'None'
        else:
            if isinstance(H, int):
                H_out = (H + pad_top + pad_bottom - kernel_h_dilated) / stride_h + 1
            else:
                H_out = 'None'
            if isinstance(W, int):
                W_out = (W + pad_right + pad_left - kernel_w_dilated) / stride_w + 1
            else:
                W_out = 'None'
    else:
        if params.isDeconvolution:
            H_out = H * stride_h if H != 'None' else 'None'
            W_out = W * stride_w if H != 'None' else 'None'
        else:
            H_out = math.ceil(H / float(stride_h)) if H != 'None' else 'None'
            W_out = math.ceil(W / float(stride_w)) if W != 'None' else 'None'

    if params.isDeconvolution:
        if len(params.outputShape) != 0:
            H_out, W_out = params.outputShape

    C_out = params.outputChannels

    N = int(N) if N != 'None' else 'None'
    C_out = int(C_out) if C_out != 'None' else 'None'
    H_out = int(H_out) if H_out != 'None' else 'None'
    W_out = int(W_out) if W_out != 'None' else 'None'

    output.type.shape = [N, C_out, H_out, W_out]


def calculate_pooling_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Pooling layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or len(input.type.shape) != 4:
        raise RuntimeError('Input must be 4-D float tensor')

    params = operator.raw_operator.pooling

    input_shape = input.type.shape
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]

    kernel_h = kernel_w = 3
    stride_h = stride_w = 1
    if len(params.kernelSize) != 0:
        kernel_h, kernel_w = params.kernelSize
    if len(params.stride) != 0:
        stride_h, stride_w = params.stride
    pad_left = pad_right = pad_bottom = pad_top = 0
    if params.globalPooling:
        H_out = W_out = 1
    else:
        if params.WhichOneof('PoolingPaddingType') == 'valid':
            if len(params.valid.paddingAmounts.borderAmounts) != 0:
                pad_top = params.valid.paddingAmounts.borderAmounts[0].startEdgeSize
                pad_bottom = params.valid.paddingAmounts.borderAmounts[0].endEdgeSize
                pad_left = params.valid.paddingAmounts.borderAmounts[1].startEdgeSize
                pad_right = params.valid.paddingAmounts.borderAmounts[1].endEdgeSize
            if H != 'None':
                H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1
            else:
                H_out = 'None'
            if W != 'None':
                W_out = (W + pad_right + pad_left - kernel_w) / stride_w + 1
            else:
                W_out = 'None'
        elif params.WhichOneof('PoolingPaddingType') == 'same':
            H_out = math.ceil(H / float(stride_h)) if H != 'None' else 'None'
            W_out = math.ceil(W / float(stride_w)) if W != 'None' else 'None'
        else:
            if len(params.includeLastPixel.paddingAmounts) != 0:
                pad_top = params.includeLastPixel.paddingAmounts[0]
                pad_bottom = pad_top
                pad_left = params.includeLastPixel.paddingAmounts[1]
                pad_right = pad_left
            if H != 'None':
                H_out = math.ceil((H + 2 * pad_top - kernel_h) / float(stride_h)) + 1
            else:
                H_out = 'None'
            if W != 'None':
                W_out = math.ceil((W + 2 * pad_left - kernel_w) / float(stride_w)) + 1
            else:
                W_out = 'None'

            if pad_top or pad_left:
                if H_out != 'None' and (H_out - 1) * stride_h >= H + pad_top:
                    H_out -= 1
                if W_out != 'None' and (W_out - 1) * stride_w >= W + pad_left:
                    W_out -= 1

    if H_out != 'None':
        H_out = int(H_out)
    if W_out != 'None':
        W_out = int(W_out)

    output.type = FloatTensorType([N, C, H_out, W_out])


def calculate_flatten_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Flatten layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or len(input.type.shape) not in [2, 4]:
        raise RuntimeError('Input must be 2-D or 4-D float tensor')

    input_shape = input.type.shape
    if len(input_shape) == 4:
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2]
        W = input_shape[3]
    elif len(input_shape) == 2:
        N = input_shape[0]
        C = input_shape[1]
        H = 1
        W = 1
    else:
        raise RuntimeError('Input shape be a 2- or 4-element list')

    if 'None' in [C, H, W]:
        C_out = 'None'
    else:
        C_out = C * H * W

    output.type = FloatTensorType([N, C_out, 1, 1])


def calculate_permute_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Flatten layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if not isinstance(input.type, TensorType) or not isinstance(output.type, TensorType):
        raise RuntimeError('Only tensor types can be permuted')

    axes = [int(i) for i in operator.raw_operator.permute.axis]
    input_shape = copy.deepcopy(input.type.shape)
    output.type = FloatTensorType([input_shape[a] for a in axes])


def calculate_lstm_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('LSTM only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D tensor')

    # Number of directions of applying LSTM
    D = 1
    # Time axis. It's equivalent to batch axis in Keras LSTM operator.
    N = input_shape[0]
    if operator.type == 'uniDirectionalLSTM':
        C = operator.raw_operator.uniDirectionalLSTM.outputVectorSize
    else:
        raise RuntimeError('Only LSTM is supported')

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = [1, C * D]
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The initial cell state of a single sequence
        Y_c_in.type.shape = [1, C * D]

    params = operator.raw_operator.uniDirectionalLSTM
    if params.params.sequenceOutput:
        operator.outputs[0].type.shape = [N, C * D]
    else:
        # This output shape should be [1, C * D] but we use [N, C * D] for back compatibility
        operator.outputs[0].type.shape = [N, C * D]
    operator.outputs[1].type.shape = [1, C * D]
    operator.outputs[2].type.shape = [1, C * D]


def calculate_bidirectional_lstm_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('Bidirectional LSTM only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    # LSTM accepts [N, C] and [N, C, 1, 1] inputs
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D tensor')

    # Time axis. It's equivalent to batch axis in Keras LSTM operator.
    N = input_shape[0]
    if operator.type == 'biDirectionalLSTM':
        C = operator.raw_operator.biDirectionalLSTM.outputVectorSize
    else:
        raise RuntimeError('Only bidirectional LSTM is supported')

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The forward initial hidden state of a single sequence
        Y_h_in.type.shape = [1, C]
        Y_h_rev_in = operator.inputs[3]  # The backward initial hidden state of a single sequence
        Y_h_rev_in.type.shape = [1, C]
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The forward initial cell state of a single sequence
        Y_c_in.type.shape = [1, C]
        Y_c_rev_in = operator.inputs[4]  # The backward initial cell state of a single sequence
        Y_c_rev_in.type.shape = [1, C]

    params = operator.raw_operator.biDirectionalLSTM
    if params.params.sequenceOutput:
        operator.outputs[0].type.shape = [N, 2 * C]
    else:
        # This output shape should be [1, C * D] but we use [N, C * D] for back compatibility
        operator.outputs[0].type.shape = [N, 2 * C]
    operator.outputs[1].type.shape = [1, C]
    operator.outputs[2].type.shape = [1, C]
    operator.outputs[3].type.shape = [1, C]
    operator.outputs[4].type.shape = [1, C]


def calculate_gru_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('GRU only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a [N, C]- or [N, C, 1, 1]-tensor')

    D = 1  # CoreML GRU is uni-directional.
    N = input_shape[0]  # Time axis. It's equivalent to batch axis in Keras LSTM operator.
    if operator.type == 'gru':
        C = operator.raw_operator.gru.outputVectorSize
    elif operator.type == 'simpleRecurrent':
        C = operator.raw_operator.simpleRecurrent.outputVectorSize
    else:
        raise RuntimeError('Only GRU and SimpleRNN are supported')

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = [1, C * D]

    if operator.raw_operator.gru.sequenceOutput:
        operator.outputs[0].type.shape = [N, C * D]
    else:
        # This output shape should be [1, C * D] but we use [N, C * D] for back compatibility
        operator.outputs[0].type.shape = [N, C * D]
    operator.outputs[1].type.shape = [1, C * D]


def calculate_embedding_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Embedding layer can only have one input and one output')

    if type(operator.inputs[0].type) not in [Int64Type, Int64TensorType]:
        raise TypeError('ONNX embedding only accepts integer input')

    input = operator.inputs[0]
    output = operator.outputs[0]

    params = operator.raw_operator.embedding

    input_shape = input.type.shape

    N = input_shape[0]

    if input_shape[1] != 1 or (len(input_shape) > 2 and (input_shape[2] != 1 or input_shape[3] != 1)):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, 1, 1, 1]')

    if len(input_shape) == 4:
        output.type.shape = [N, params.outputChannels, 1, 1]
    elif len(input_shape) == 2:
        output.type.shape = [N, params.outputChannels]
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')


def calculate_concat_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('At least one input variable is required')
    if len(operator.outputs) > 1:
        raise RuntimeError('Only one output variable can be produced')

    input_shape = operator.inputs[0].type.shape
    N = input_shape[0]
    C = []
    H = input_shape[2]
    W = input_shape[3]

    for variable in operator.inputs:
        if variable.type.shape[0] != 'None' and variable.type.shape[0] != N:
            raise RuntimeError('Only dimensions along C-axis can be different')
        if variable.type.shape[2] != 'None' and variable.type.shape[2] != H:
            raise RuntimeError('Only dimensions along C-axis can be different')
        if variable.type.shape[3] != 'None' and variable.type.shape[3] != W:
            raise RuntimeError('Only dimensions along C-axis can be different')
        C.append(variable.type.shape[1])

    C = 'None' if 'None' in C else sum(C)

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculte_tensor_to_label_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-label operator has only one input and output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

    N = operator.inputs[0].type.shape[0]
    if type(operator.outputs[0].type) == Int64Type:
        if N == 1:
            operator.outputs[0].type = Int64Type()
        else:
            operator.outputs[0].type = Int64TensorType([N, 1])
    elif type(operator.outputs[0].type) == StringType:
        if N == 1:
            operator.outputs[0].type = StringTensorType([N, 1])
        else:
            operator.outputs[0].type = StringType()
    else:
        raise RuntimeError('Unsupported label type')


def calculate_tensor_to_probability_map_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-label operator has only one input and output')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        class_label_type = operator.raw_operator.neuralNetworkClassifier.WhichOneof('ClassLabels')
    else:
        raise TypeError('%s has no class label' % model_type)

    N = operator.inputs[0].type.shape[0]
    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = DictionaryType(StringType(), FloatTensorType([1]))
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(StringType(), FloatType()), N)
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = DictionaryType(Int64Type(), FloatTensorType([1]))
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(Int64Type(), FloatType()), N)
    else:
        raise TypeError('Unsupported label type')


def calculate_reshape_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reshape operator has only one input and output')

    params = operator.raw_operator.reshape

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Only float tensors can be reshaped')

    N = operator.inputs[0].type.shape[0]

    if len(params.targetShape) == 3:
        C, H, W = params.targetShape
    else:
        N, C, H, W = params.targetShape

    N = int(N) if N != 'None' else 'None'
    C = int(C) if C != 'None' else 'None'
    H = int(H) if H != 'None' else 'None'
    W = int(W) if W != 'None' else 'None'

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_dictionary_vectorizer_output_shapes(operator):
    # We assume all dictionaries' value types are float. It seems be reasonable to CoreML's
    # model input, but the existence of other map types leads to some concerns.
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Dictionary vectorizer operator has only one input and output')

    # [TODO] dictionary vectorizer should be able to accept a sequence of dictionary
    if type(operator.inputs[0].type) != DictionaryType and type(operator.inputs[0].type) != SequenceType:
        raise TypeError('Input type must be a sequence of dictionary or a dictionary of a sequence')

    params = operator.raw_operator.dictVectorizer
    string_key_vector = params.stringToIndex.vector
    int64_key_vector = params.int64ToIndex.vector

    if len(string_key_vector) > 0 and len(int64_key_vector) > 0:
        raise RuntimeError('Only one key type can present at the same time')

    if len(string_key_vector) > 0:
        operator.outputs[0].type = FloatTensorType([1, len(string_key_vector)])
    elif len(int64_key_vector) > 0:
        operator.outputs[1].type = FloatTensorType([1, len(int64_key_vector)])
    else:
        raise RuntimeError('Key vector cannot be empty')


def calculate_feature_vectorizer_output_shapes(operator):
    if len(operator.outputs) != 1:
        raise RuntimeError('Feature vectorizer operator has only one output')
    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be float or integer tensor(s)')
    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be 2-D tensor(s)')

    # Find the first batch size which is not unknown
    N = 'None'
    for variable in operator.inputs:
        if variable.type.shape[0] != 'None':
            N = variable.type.shape[0]
            break
    for variable in operator.inputs:
        if variable.type.shape[0] not in ['None', N]:
            raise RuntimeError('The batch dimensions should be the same to all input tensors.')

    C = sum(info.inputDimensions for info in operator.raw_operator.featureVectorizer.inputList)

    if isinstance(operator.inputs[0].type, (FloatTensorType, FloatType)):
        operator.outputs[0].type = FloatTensorType([N, C])
    elif isinstance(operator.inputs[0].type, (Int64TensorType, Int64Type)):
        operator.outputs[0].type = Int64TensorType([N, C])
    else:
        raise RuntimeError('Unsupported input type: %s' % type(operator.inputs[0].type))


def calculate_traditional_classifier_output_shapes(operator):
    if len(operator.inputs) != 1:
        raise RuntimeError('Classifier has only one input')
    if len(operator.outputs) > 2 or len(operator.outputs) < 1:
        raise RuntimeError('Classifier cannot produce more than two or zero output')

    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be tensor(s)')
    if any(len(variable.type.shape) != 2 or variable.type.shape[0] != 1 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be [1,C]-tensor(s)')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'treeEnsembleClassifier':
        class_label_type = operator.raw_operator.treeEnsembleClassifier.WhichOneof('ClassLabels')
    elif model_type == 'glmClassifier':
        class_label_type = operator.raw_operator.glmClassifier.WhichOneof('ClassLabels')
    elif model_type == 'supportVectorClassifier':
        class_label_type = operator.raw_operator.supportVectorClassifier.WhichOneof('ClassLabels')
    else:
        raise TypeError('%s has no class label' % model_type)

    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = StringType()
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(StringType(), FloatType())
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = Int64Type()
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(Int64Type(), FloatType())
    else:
        raise ValueError('Traditional classifier must include label information')


def calculate_traditional_regressor_output_shapes(operator):
    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType, FloatType, Int64Type)) for variable in
           operator.inputs):
        raise RuntimeError('Input(s) must be tensor(s)')
    if any(len(variable.type.shape) != 2 for variable in operator.inputs):
        raise RuntimeError('Input(s) must be 2-D tensor(s)')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'glmRegressor':
        glm = operator.raw_operator.glmRegressor
        C = len(glm.weights)
    elif model_type == 'treeEnsembleRegressor':
        tree = operator.raw_operator.treeEnsembleRegressor.treeEnsemble
        C = len(tree.basePredictionValue)
    elif model_type == 'supportVectorRegressor':
        C = 1
    else:
        raise ValueError('Model should be one of linear model, tree-based model, and support vector machine')

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, C])


def calculate_array_feature_extractor_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tree ensemble classifier has only one input and one output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

    N = operator.inputs[0].type.shape[0]
    extracted_feature_number = len(operator.raw_operator.arrayFeatureExtractor.extractIndex)

    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, extracted_feature_number]


def calculate_one_hot_encoder_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('One-hot encoder has only one input and one output')
    if operator.inputs[0].type.shape[1] != 1 or len(operator.inputs[0].type.shape) > 2:
        raise RuntimeError('Input must be [N, 1]-tensor')

    int_categories = operator.raw_operator.oneHotEncoder.int64Categories.vector
    str_categories = operator.raw_operator.oneHotEncoder.stringCategories.vector

    N = operator.inputs[0].type.shape[0]

    if len(int_categories) > 0:
        operator.outputs[0].type = FloatTensorType([N, len(int_categories)])
    elif len(str_categories) > 0 and type(operator.inputs[0].type) == StringTensorType:
        operator.outputs[0].type = FloatTensorType([N, len(str_categories)])
    else:
        raise RuntimeError('Categorical indexes are missing')


def calculate_padding_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Padding is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a tensor')

    params = operator.raw_operator.padding
    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]
    l = r = t = b = 0
    if len(params.paddingAmounts.borderAmounts) != 0:
        t = params.paddingAmounts.borderAmounts[0].startEdgeSize
        b = params.paddingAmounts.borderAmounts[0].endEdgeSize
        l = params.paddingAmounts.borderAmounts[1].startEdgeSize
        r = params.paddingAmounts.borderAmounts[1].endEdgeSize

    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, C, H + t + b, W + l + r]


def calculate_batch_normalization_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Batch normalization is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    input_shape = operator.inputs[0].type.shape
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)


def calculate_crop_output_shapes(operator):
    if len(operator.inputs) > 2 or len(operator.outputs) != 1:
        raise RuntimeError('Invalid input or output numbers')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    params = operator.raw_operator.crop
    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]
    pad_left = pad_right = pad_top = pad_bottom = 0
    if len(operator.inputs) == 1:
        if len(params.cropAmounts.borderAmounts) != 0:
            pad_top = params.cropAmounts.borderAmounts[0].startEdgeSize
            pad_bottom = params.cropAmounts.borderAmounts[0].endEdgeSize
            pad_left = params.cropAmounts.borderAmounts[1].startEdgeSize
            pad_right = params.cropAmounts.borderAmounts[1].endEdgeSize
        H = H - pad_top - pad_bottom
        W = W - pad_left - pad_right
    else:
        H = operator.raw_operator.inputs[1].type.shape[3]
        W = operator.raw_operator.inputs[1].type.shape[4]

    operator.outputs[0].type.shape = [N, C, H, W]


def calculate_add_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('Add operator requires at least one input')
    if len(operator.outputs) != 1:
        raise RuntimeError('Add operator only has one output')

    for variable in operator.inputs:
        if not isinstance(variable.type, FloatTensorType):
            raise RuntimeError('Input must be a float tensor')

    # [TODO] Fix reduce-like shape inference. We now assume all inputs are 4-D.

    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]

    for variable in operator.inputs:
        C_new = variable.type.shape[1]
        H_new = variable.type.shape[2]
        W_new = variable.type.shape[3]
        if C != C_new and 1 not in [C, C_new]:
            raise RuntimeError('Bad shapes (%s, %s) for broadcasting' % (C, C_new))
        C = max(C, C_new)
        if H != H_new and 1 not in [H, H_new]:
            raise RuntimeError('Bad shape (%s, %s) for broadcasting' % (H, H_new))
        H = max(H, H_new)
        if W != W_new and 1 not in [W, W_new]:
            raise RuntimeError('Bad shape (%s, %s) for broadcasting' % (W, W_new))
        W = max(W, W_new)

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_upsample_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')
    scales = operator.raw_operator.upsample.scalingFactor

    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2] * scales[0]
    W = operator.inputs[0].type.shape[3] * scales[1]

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_split_output_shapes(operator):
    if len(operator.inputs) != 1:
        raise RuntimeError('Split has only one input')

    if len(operator.inputs) < 1:
        raise RuntimeError('Split should create at least one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    input_shape = operator.inputs[0].type.shape
    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]

    split_num = operator.raw_operator.split.nOutputs

    if C % split_num != 0:
        raise RuntimeError(
            'Split Operator, %s, got wrong input shape. Variable %s''s dimension along C-axis (%s) must be divisible by partition number (%s)' % (
                operator.full_name, operator.inputs[0].full_name, C, split_num))

    if len(input_shape) == 4:
        operator.outputs[0].type = FloatTensorType([N, int(C / split_num), input_shape[2], input_shape[3]])
    elif len(input_shape) == 2:
        operator.outputs[0].type = FloatTensorType([N, int(C / split_num), 1, 1])
    else:
        raise RuntimeError('Input must be a 2-D or 4-D tensor')


def calculate_slice_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]

    params = operator.raw_operator.slice

    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params
    axis_map = {Params.CHANNEL_AXIS: 0, Params.HEIGHT_AXIS: 1, Params.WIDTH_AXIS: 2}
    if params.axis == Params.CHANNEL_AXIS:
        if params.endIndex >= 0:
            C = params.endIndex - params.startIndex
        else:
            C = (C + 1 + params.endIndex) - params.startIndex
    elif params.axis == Params.HEIGHT_AXIS:
        if params.endIndex >= 0:
            H = params.endIndex - params.startIndex
        else:
            H = (H + 1 + params.endIndex) - params.startIndex
    elif params.axis == Params.WIDTH_AXIS:
        if params.endIndex >= 0:
            W = params.endIndex - params.startIndex
        else:
            W = (W + 1 + params.endIndex) - params.startIndex
    else:
        raise RuntimeError('Unsupported slicing axis')

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_sequence_repeat_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    repeat_num = operator.raw_operator.sequenceRepeat.nRepetitions

    N = operator.inputs[0].type.shape[0] * repeat_num
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_reorganizeData_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]

    params = operator.raw_operator.reorganizeData

    from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params
    if params.mode == Params.SPACE_TO_DEPTH:
        if H % params.blockSize != 0 or W % params.blockSize != 0:
            raise RuntimeError('Height and weight must be divisible by block size')

        C = int(C * params.blockSize * params.blockSize)
        H = int(H / params.blockSize)
        W = int(W / params.blockSize)
    elif params.mode == Params.DEPTH_TO_SPACE:
        if C % (params.blockSize * params.blockSize) != 0:
            raise RuntimeError('Channel number must be divisible by the square of block size')

        C = int(C / params.blockSize / params.blockSize)
        H = int(H * params.blockSize)
        W = int(W * params.blockSize)
    else:
        raise ValueError('Unsupport reorganization mode {0}'.format(params.mode))

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_reduce_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    N = operator.inputs[0].type.shape[0]
    C = operator.inputs[0].type.shape[1]
    H = operator.inputs[0].type.shape[2]
    W = operator.inputs[0].type.shape[3]

    params = operator.raw_operator.reduce

    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params
    if params.axis == Params.CHW:
        C = 1
        H = 1
        W = 1
    elif params.axis == Params.HW:
        H = 1
        W = 1
    elif params.axis == Params.C:
        C = 1
    elif params.axis == Params.H:
        H = 1
    elif params.axis == Params.W:
        W = 1
    else:
        raise RuntimeError('Unsupported reduction mode')

    operator.outputs[0].type = FloatTensorType([N, C, H, W])


def calculate_load_constant_output_shapes(operator):
    if len(operator.inputs) != 0:
        raise RuntimeError('LoadConstant operator has no input')
    if len(operator.outputs) != 1:
        raise RuntimeError('LoadConstant operator has only one output')
    output = operator.outputs[0]

    # CoreML's constant is always 3-D tensor, so we assume its shape is [C, H, W].
    const_shape = operator.raw_operator.loadConstant.shape
    # We convert [C, H, W] to [1, C, H, W] because our parsing code use [N, C, H, W] as the inferface between operators.
    const_shape = [1] + [int(d) for d in const_shape]
    if output.type is None:
        # Use default type
        output.type = FloatTensorType(shape=const_shape)
    else:
        if not isinstance(output.type, TensorType):
            raise RuntimeError('Type conflict detected. Output must be a tensor.')
        # If output type exists, we just modify its shape.
        output.type.shape = const_shape


def calculate_dot_output_shapes(operator):
    if len(operator.inputs) != 2 or len(operator.outputs) != 1:
        raise RuntimeError('Dot must have two inputs and one output')
    if any(type(variable.type) != FloatTensorType for variable in operator.inputs):
        raise RuntimeError('Input(s) must be float tensor(s)')
    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        raise RuntimeError('Input shapes must be identical')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) == 2:
        operator.outputs[0].type = FloatTensorType([input_shape[0], 1])
    else:
        raise RuntimeError('Input(s) must be a 2-D tensor(s)')


# [TODO] Support 2-D tesnor as [N, C, 1, 1] in neural network's shape calculators
type_calculator_table = {'activation': calculate_activation_output_shapes,
                         'innerProduct': calculate_inner_product_output_shapes,
                         'identity': calculate_identity_output_shapes,
                         'softmax': calculate_softmax_output_shapes,
                         'convolution': calculate_convolution_output_shapes,
                         'pooling': calculate_pooling_output_shapes,
                         'scalerPreprocessor': calculate_softmax_output_shapes,
                         'flatten': calculate_flatten_output_shapes,
                         'permute': calculate_permute_output_shapes,
                         'imageToFloatTensor': calculate_image_to_float_tensor_output_shapes,
                         'uniDirectionalLSTM': calculate_lstm_output_shapes,
                         'biDirectionalLSTM': calculate_bidirectional_lstm_output_shapes,
                         'embedding': calculate_embedding_output_shapes,
                         'concat': calculate_concat_output_shapes,
                         'reshape': calculate_reshape_output_shapes,
                         'tensorToLabel': calculte_tensor_to_label_output_shapes,
                         'tensorToProbabilityMap': calculate_tensor_to_probability_map_output_shapes,
                         'gru': calculate_gru_output_shapes,
                         'dictVectorizer': calculate_dictionary_vectorizer_output_shapes,
                         'featureVectorizer': calculate_feature_vectorizer_output_shapes,
                         'treeEnsembleClassifier': calculate_traditional_classifier_output_shapes,
                         'glmClassifier': calculate_traditional_classifier_output_shapes,
                         'glmRegressor': calculate_traditional_regressor_output_shapes,
                         'arrayFeatureExtractor': calculate_array_feature_extractor_output_shapes,
                         'oneHotEncoder': calculate_one_hot_encoder_output_shapes,
                         'padding': calculate_padding_output_shapes,
                         'batchnorm': calculate_batch_normalization_output_shapes,
                         'crop': calculate_crop_output_shapes,
                         'add': calculate_add_output_shapes,
                         'scale': calculate_identity_output_shapes,
                         'upsample': calculate_upsample_output_shapes,
                         'unary': calculate_softmax_output_shapes,
                         'split': calculate_split_output_shapes,
                         'slice': calculate_slice_output_shapes,
                         'simpleRecurrent': calculate_gru_output_shapes,
                         'sequenceRepeat': calculate_sequence_repeat_output_shapes,
                         'reorganizeData': calculate_reorganizeData_output_shapes,
                         'reduce': calculate_reduce_output_shapes,
                         'multiply': calculate_add_output_shapes,
                         'min': calculate_add_output_shapes,
                         'mvn': calculate_identity_output_shapes,
                         'max': calculate_add_output_shapes,
                         'lrn': calculate_identity_output_shapes,
                         'loadConstant': calculate_load_constant_output_shapes,
                         'l2normalize': calculate_identity_output_shapes,
                         'dot': calculate_dot_output_shapes,
                         'bias': calculate_identity_output_shapes,
                         'average': calculate_add_output_shapes,
                         'supportVectorRegressor': calculate_traditional_regressor_output_shapes,
                         'supportVectorClassifier': calculate_traditional_classifier_output_shapes,
                         'scaler': calculate_identity_output_shapes,
                         'treeEnsembleClassifier': calculate_traditional_classifier_output_shapes,
                         'treeEnsembleRegressor': calculate_traditional_regressor_output_shapes,
                         'imputer': calculate_identity_output_shapes,
                         'normalizer': calculate_softmax_output_shapes
                         }
