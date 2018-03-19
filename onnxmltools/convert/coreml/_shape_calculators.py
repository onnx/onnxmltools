#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import copy
import math
from ._data_types import *


def calculate_inner_product_output_shapes(operator):
    # Input shape: [N, C]- or [N, C, 1, 1]-tensor
    # Output shape: [N, C']- or [N, C', 1, 1]-tensor
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Inner product layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType:
        raise RuntimeError('Input must be float tensor')

    input_shape = input.type.shape
    if len(input_shape) == 4 and (input_shape[2] != 1 or input_shape[3] != 1):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, C, 1, 1]')

    params = operator.raw_operator.innerProduct

    if input_shape[1] != params.inputChannels:
        raise RuntimeError('Dimension mismatch along C-axis. Expected %s but got %s' %
                           (params.inputChannels, input_shape[1]))

    if len(input_shape) == 4:
        output.type.shape = [input_shape[0], params.outputChannels, 1, 1]
    elif len(input_shape) == 2:
        output.type.shape = [input_shape[0], params.outputChannels]
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')


def calculate_identical_float_tensor_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('This layer %s can only have one input and one output' % operator.type)

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or type(output.type) != FloatTensorType:
        raise RuntimeError('Input must be float tensor')

    doc_string = output.type.doc_string
    output.type.shape = copy.deepcopy(input.type.shape)  # Similar to identity but only accept floats
    output.type.doc_string = doc_string


def calculate_identity_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Identity layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    doc_string = output.type.doc_string
    output.type = copy.deepcopy(input.type)
    output.type.doc_string = doc_string


def calculate_convolution_and_pooling_1D_output_shape(
        input_size, kernel_size, kernel_dilation, stride, pad_mode, pad_head, pad_tail, output_size=0):
    if not isinstance(input_size, numbers.Integral):
        return 'None'
    if output_size > 0:
        return int(output_size)  # Must use output_size = 1 for global pooling

    effective_kernel_size = 1 + kernel_dilation * (kernel_size - 1)  # For pooling, we always have dilation = 1.
    if pad_mode == 'valid':
        return int(math.floor((input_size + pad_head + pad_tail - effective_kernel_size) / stride) + 1)
    elif pad_mode == 'same':
        return int(math.ceil(input_size / stride))
    elif pad_mode == 'includeLastPixel':
        if pad_head != pad_tail:
            raise ValueError('Padding amounts at the beginning and the end of an axis must be the same')
        effective_input_size = input_size + pad_head + pad_tail - effective_kernel_size
        out_size = math.ceil(effective_input_size / stride) + 1
        if (out_size - 1) * stride >= input_size + pad_head:
            out_size -= 1
        return out_size
    else:
        raise ValueError('Unknown padding mode: %s' % pad_mode)


def calculate_convolution_transpose_1D_output_shape(
        input_size, kernel_size, kernel_dilation, stride, pad_mode, pad_head, pad_tail, output_size=0):
    if not isinstance(input_size, numbers.Integral):
        return 'None'
    if output_size > 0:
        return output_size

    effective_kernel_size = 1 + kernel_dilation * (kernel_size - 1)
    if pad_mode == 'valid':
        return int((input_size - 1) * stride - pad_head - pad_tail + effective_kernel_size)
    elif pad_mode == 'same':
        return int(input_size * stride)
    else:
        raise ValueError('Unknown padding mode: %s' % pad_mode)


def calculate_convolution_output_shapes(operator):
    params = operator.raw_operator.convolution

    input_shape = operator.inputs[0].type.shape
    operator.outputs[0].type.shape = [0, 0, 0, 0]  # Initialize output shape. It will be modified below.
    output_shape = operator.outputs[0].type.shape

    # Adjust N-axis
    output_shape[0] = input_shape[0]

    # Adjust C-axis
    output_shape[1] = params.outputChannels

    # Set up default and non-default parameters
    dilations = [1, 1]
    if len(params.dilationFactor) > 0:
        dilations = [params.dilationFactor[0], params.dilationFactor[1]]
    kernel_shape = [3, 3]
    if len(params.kernelSize) > 0:
        kernel_shape = params.kernelSize
    strides = [1, 1]
    if len(params.stride) > 0:
        strides = params.stride
    specified_output_shape = [0, 0]  # Only used with convolution transpose
    if params.isDeconvolution and len(params.outputShape) > 0:
        specified_output_shape = list(int(i) for i in params.outputShape)
    pad_mode = params.WhichOneof('ConvolutionPaddingType')
    if pad_mode == 'valid' and len(params.valid.paddingAmounts.borderAmounts) > 0:
        pad_amounts = params.valid.paddingAmounts.borderAmounts
        pad_heads = [pad_amounts[0].startEdgeSize, pad_amounts[1].startEdgeSize]
        pad_tails = [pad_amounts[0].endEdgeSize, pad_amounts[1].endEdgeSize]
    else:
        # Padding amounts are useless for same padding and valid padding uses [0, 0] by default.
        pad_heads = [0, 0]
        pad_tails = [0, 0]

    # Adjust H- and W-axes
    for i in range(2):
        if params.isDeconvolution:
            output_shape[i + 2] = calculate_convolution_transpose_1D_output_shape(
                input_shape[i + 2], kernel_shape[i], dilations[i], strides[i],
                pad_mode, pad_heads[i], pad_tails[i], specified_output_shape[i])
        else:
            output_shape[i + 2] = calculate_convolution_and_pooling_1D_output_shape(
                input_shape[i + 2], kernel_shape[i], dilations[i], strides[i],
                pad_mode, pad_heads[i], pad_tails[i])


def calculate_pooling_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Pooling layer can only have one input and one output')

    input = operator.inputs[0]
    input_shape = operator.inputs[0].type.shape

    if type(input.type) != FloatTensorType or len(input.type.shape) != 4:
        raise RuntimeError('Input must be 4-D float tensor')

    operator.outputs[0].type.shape = [0, 0, 0, 0]
    output_shape = operator.outputs[0].type.shape

    # Adjust N-axis
    output_shape[0] = input_shape[0]

    # Adjust C-axis
    output_shape[1] = input_shape[1]

    params = operator.raw_operator.pooling
    # Set up default and non-default parameters. Notice that they are only set for H- and W-axes.
    dilations = [1, 1]  # CoreML Pooling doesn't allow dilation, so we use [1, 1] which is equivalent to no dilation.
    kernel_shape = [3, 3]
    if len(params.kernelSize) > 0:
        kernel_shape = params.kernelSize
    strides = [1, 1]
    if len(params.stride) > 0:
        strides = params.stride
    pad_mode = params.WhichOneof('PoolingPaddingType')
    if pad_mode == 'valid' and len(params.valid.paddingAmounts.borderAmounts) > 0:
        pad_amounts = params.valid.paddingAmounts.borderAmounts
        pad_heads = [pad_amounts[0].startEdgeSize, pad_amounts[1].startEdgeSize]
        pad_tails = [pad_amounts[0].endEdgeSize, pad_amounts[1].endEdgeSize]
    elif pad_mode == 'includeLastPixel' and len(params.includeLastPixel.paddingAmounts) > 0:
        pad_amounts = params.includeLastPixel.paddingAmounts
        pad_heads = [pad_amounts[0], pad_amounts[1]]
        pad_tails = [pad_amounts[0], pad_amounts[1]]
    else:
        # For same padding, padding amounts are not used
        pad_heads = [0, 0]
        pad_tails = [0, 0]

    # Calculate output shape along H- and W-axes
    for i in range(2):
        output_shape[i + 2] = calculate_convolution_and_pooling_1D_output_shape(
            input_shape[i + 2], kernel_shape[i], dilations[i], strides[i],
            pad_mode, pad_heads[i], pad_tails[i], params.globalPooling)


def calculate_flatten_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Flatten layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or len(input.type.shape) not in [2, 4]:
        raise RuntimeError('Input must be 2-D or 4-D float tensor')

    input_shape = input.type.shape
    output_shape = [input_shape[0], 1, 1, 1]

    # Calculate the multiplication of C, H, and W.
    for i in input_shape[1:]:
        if i != 'None':
            output_shape[1] *= i
        else:
            # If any of C, H, W-dimensions is unknown, the flatten C-dimension is unknown
            output_shape[1] = 'None'
            break

    output.type.shape = output_shape


def calculate_permute_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Permute layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if not isinstance(input.type, TensorType) or not isinstance(output.type, TensorType):
        raise RuntimeError('Only tensor types can be permuted')

    axes = [int(i) for i in operator.raw_operator.permute.axis]
    input_shape = copy.deepcopy(input.type.shape)
    output.type.shape = [input_shape[a] for a in axes]


def calculate_lstm_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('LSTM only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D tensor')

    params = operator.raw_operator.uniDirectionalLSTM

    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [1, params.outputVectorSize]
    output_shape = ['None', params.outputVectorSize]
    state_shape = [1, params.outputVectorSize]

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The initial cell state of a single sequence
        Y_c_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape
    if len(operator.outputs) > 2:
        operator.outputs[2].type.shape = state_shape


def calculate_bidirectional_lstm_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('Bidirectional LSTM only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    # LSTM accepts [N, C] and [N, C, 1, 1] inputs
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or 4-D tensor')

    params = operator.raw_operator.biDirectionalLSTM
    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [2, params.outputVectorSize]
    output_shape = ['None', params.outputVectorSize]
    state_shape = [1, params.outputVectorSize]

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The forward initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape
        Y_h_rev_in = operator.inputs[3]  # The backward initial hidden state of a single sequence
        Y_h_rev_in.type.shape = state_shape
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The forward initial cell state of a single sequence
        Y_c_in.type.shape = state_shape
        Y_c_rev_in = operator.inputs[4]  # The backward initial cell state of a single sequence
        Y_c_rev_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape
        operator.outputs[3].type.shape = state_shape
    if len(operator.outputs) > 2:
        operator.outputs[2].type.shape = state_shape
        operator.outputs[4].type.shape = state_shape


def calculate_gru_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('GRU only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a [N, C]- or [N, C, 1, 1]-tensor')

    if operator.type == 'gru':
        params = operator.raw_operator.gru
    elif operator.type == 'simpleRecurrent':
        params = operator.raw_operator.simpleRecurrent
    else:
        raise RuntimeError('Only GRU and SimpleRNN are supported')

    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [2, params.outputVectorSize]
    output_shape = [input_shape[0] if params.sequenceOutput else 'None', params.outputVectorSize] # 'None' should be 1
    state_shape = [1, params.outputVectorSize]

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape


def calculate_embedding_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Embedding layer can only have one input and one output')

    if type(operator.inputs[0].type) not in [Int64Type, Int64TensorType]:
        raise RuntimeError('ONNX embedding only accepts integer input')

    output = operator.outputs[0]

    input_shape = operator.inputs[0].type.shape

    if input_shape[1] != 1 or (len(input_shape) > 2 and (input_shape[2] != 1 or input_shape[3] != 1)):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, 1, 1, 1]')

    params = operator.raw_operator.embedding
    if len(input_shape) == 4:
        output_shape = [input_shape[0], params.outputChannels, 1, 1]
    elif len(input_shape) == 2:
        output_shape = [input_shape[0], params.outputChannels]
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    output.type.shape = output_shape


def calculate_concat_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('At least one input variable is required')
    if len(operator.outputs) > 1:
        raise RuntimeError('Only one output variable can be produced')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    dims = []
    for variable in operator.inputs:
        if variable.type.shape[0] != 'None' and variable.type.shape[0] != output_shape[0]:
            raise RuntimeError('Only dimensions along C-axis can be different')
        if variable.type.shape[2] != 'None' and variable.type.shape[2] != output_shape[2]:
            raise RuntimeError('Only dimensions along C-axis can be different')
        if variable.type.shape[3] != 'None' and variable.type.shape[3] != output_shape[3]:
            raise RuntimeError('Only dimensions along C-axis can be different')
        dims.append(variable.type.shape[1])

    output_shape[1] = 'None' if 'None' in dims else sum(dims)
    operator.outputs[0].type.shape = output_shape


def calculte_tensor_to_label_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-label operator has only one input and output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

    N = operator.inputs[0].type.shape[0]
    if type(operator.outputs[0].type) == Int64Type:
        operator.outputs[0].type = Int64TensorType([1], doc_string=operator.outputs[0].type.doc_string)
        # Due to the limitation of ZipMap, we are not able to produce label and class probability map for batch size
        # greater than 1. It leads to that although the following code is semantically correct, we cannot use it.
        # if N == 1:
        #    operator.outputs[0].type = Int64Type()
        # else:
        #    operator.outputs[0].type = Int64TensorType([N, 1])
    elif type(operator.outputs[0].type) == StringType:
        operator.outputs[0].type = StringTensorType([1], doc_string=operator.outputs[0].type.doc_string)
        # Due to the limitation of ZipMap, we are not able to produce label and class probability map for batch size
        # greater than 1. It leads to that although the following code is semantically correct, we cannot use it.
        # if N == 1:
        #    operator.outputs[0].type = StringTensorType([N, 1])
        # else:
        #    operator.outputs[0].type = StringType()
    else:
        raise ValueError('Unsupported label type')


def calculate_tensor_to_probability_map_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Tensor-to-probability operator has only one input and output')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    model_type = operator.raw_operator.WhichOneof('Type')
    if model_type == 'neuralNetworkClassifier':
        class_label_type = operator.raw_operator.neuralNetworkClassifier.WhichOneof('ClassLabels')
    else:
        raise TypeError('%s has no class label' % model_type)

    N = operator.inputs[0].type.shape[0]
    doc_string = operator.outputs[0].type.doc_string
    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = DictionaryType(StringType(), FloatTensorType([1]), doc_string=doc_string)
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(StringType(), FloatType()), N)
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = DictionaryType(Int64Type(), FloatTensorType([1]), doc_string=doc_string)
        # It should be a sequence of dictionary if batch size is larger than 1, but runtime don't have such a type.
        # operator.outputs[0].type = SequenceType(DictionaryType(Int64Type(), FloatType()), N)
    else:
        raise ValueError('Unsupported label type')


def calculate_reshape_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reshape operator has only one input and output')

    params = operator.raw_operator.reshape

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Only float tensors can be reshaped')

    output_shape = list(int(i) for i in params.targetShape)

    if len(output_shape) == 3:
        output_shape = [operator.inputs[0].type.shape[0]] + output_shape

    operator.outputs[0].type.shape = output_shape


def calculate_dictionary_vectorizer_output_shapes(operator):
    # We assume all dictionaries' value types are float. It seems be reasonable to CoreML's
    # model input, but the existence of other map types leads to some concerns.
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Dictionary vectorizer operator has only one input and output')

    # [TODO] dictionary vectorizer should be able to accept a sequence of dictionary
    if type(operator.inputs[0].type) != DictionaryType and type(operator.inputs[0].type) != SequenceType:
        raise RuntimeError('Input type must be a sequence of dictionary or a dictionary of a sequence')

    params = operator.raw_operator.dictVectorizer
    string_key_vector = params.stringToIndex.vector
    int64_key_vector = params.int64ToIndex.vector

    if len(string_key_vector) > 0 and len(int64_key_vector) > 0:
        raise RuntimeError('Only one key type can present at the same time')

    doc_string = operator.outputs[0].type.doc_string
    if len(string_key_vector) > 0:
        operator.outputs[0].type = FloatTensorType([1, len(string_key_vector)], doc_string=doc_string)
    elif len(int64_key_vector) > 0:
        operator.outputs[1].type.shape = FloatTensorType([1, len(int64_key_vector)], doc_string=doc_string)
    else:
        raise ValueError('Key vector cannot be empty')


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
        doc_string = operator.outputs[0].type.doc_string
        operator.outputs[0].type = FloatTensorType([N, C], doc_string=doc_string)
    elif isinstance(operator.inputs[0].type, (Int64TensorType, Int64Type)):
        doc_string = operator.outputs[0].type.doc_string
        operator.outputs[0].type = Int64TensorType([N, C], doc_string=doc_string)
    else:
        raise ValueError('Unsupported input type: %s' % type(operator.inputs[0].type))


def calculate_traditional_classifier_output_shapes(operator):
    if len(operator.outputs) > 2 or len(operator.outputs) < 1:
        raise RuntimeError('Classifier cannot produce more than two or no output')

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
        raise ValueError('%s has no class label' % model_type)

    if class_label_type == 'stringClassLabels':
        operator.outputs[0].type = StringType(doc_string=operator.outputs[0].type.doc_string)
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(StringType(), FloatType(),
                                                      doc_string=operator.outputs[1].type.doc_string)
    elif class_label_type == 'int64ClassLabels':
        operator.outputs[0].type = Int64Type(doc_string=operator.outputs[0].type.doc_string)
        if len(operator.outputs) == 2:
            operator.outputs[1].type = DictionaryType(Int64Type(), FloatType(),
                                                      doc_string=operator.outputs[1].type.doc_string)
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
    operator.outputs[0].type = FloatTensorType([N, C], doc_string=operator.outputs[0].type.doc_string)


def calculate_array_feature_extractor_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Array feature extractor has only one input and one output')

    if not isinstance(operator.inputs[0].type, TensorType):
        raise RuntimeError('Input must be a tensor')

    N = operator.inputs[0].type.shape[0]
    extracted_feature_number = len(operator.raw_operator.arrayFeatureExtractor.extractIndex)

    # Save doc_string before over-writing by us
    doc_string = operator.outputs[0].type.doc_string
    operator.outputs[0].type = copy.deepcopy(operator.inputs[0].type)
    operator.outputs[0].type.shape = [N, extracted_feature_number]
    # Assign correct doc_string to the output
    operator.outputs[0].type.doc_string = doc_string



def calculate_one_hot_encoder_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('One-hot encoder has only one input and one output')
    if operator.inputs[0].type.shape[1] != 1 or len(operator.inputs[0].type.shape) > 2:
        raise RuntimeError('Input must be [N, 1]-tensor')

    int_categories = operator.raw_operator.oneHotEncoder.int64Categories.vector
    str_categories = operator.raw_operator.oneHotEncoder.stringCategories.vector

    N = operator.inputs[0].type.shape[0]

    if len(int_categories) > 0:
        operator.outputs[0].type = FloatTensorType([N, len(int_categories)],
                                                   doc_string=operator.outputs[0].type.doc_string)
    elif len(str_categories) > 0 and type(operator.inputs[0].type) == StringTensorType:
        operator.outputs[0].type = FloatTensorType([N, len(str_categories)],
                                                   doc_string=operator.outputs[0].type.doc_string)
    else:
        raise ValueError('Categorical indexes are missing')


def calculate_padding_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Padding is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.padding
    if len(params.paddingAmounts.borderAmounts) > 0:
        output_shape[2] += params.paddingAmounts.borderAmounts[0].startEdgeSize
        output_shape[2] += params.paddingAmounts.borderAmounts[0].endEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].startEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].endEdgeSize

    operator.outputs[0].type.shape = output_shape


def calculate_batch_normalization_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Batch normalization is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    input_shape = operator.inputs[0].type.shape
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    operator.outputs[0].type.shape = copy.deepcopy(operator.inputs[0].type.shape)


def calculate_crop_output_shapes(operator):
    if len(operator.inputs) > 2 or len(operator.outputs) != 1:
        raise RuntimeError('Invalid input or output numbers')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.crop
    if len(operator.inputs) == 1:
        if len(params.cropAmounts.borderAmounts) > 0:
            output_shape[2] -= params.cropAmounts.borderAmounts[0].startEdgeSize
            output_shape[2] -= params.cropAmounts.borderAmounts[0].endEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].startEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].endEdgeSize
    elif len(operator.inputs) == 2:
        output_shape[2] = operator.raw_operator.inputs[1].type.shape[2]
        output_shape[3] = operator.raw_operator.inputs[1].type.shape[3]
    else:
        raise RuntimeError('Too many inputs for Crop operator')

    operator.outputs[0].type.shape = output_shape


def calculate_add_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('Add operator requires at least one input')
    if len(operator.outputs) != 1:
        raise RuntimeError('Add operator only has one output')

    for variable in operator.inputs:
        if not isinstance(variable.type, FloatTensorType):
            raise RuntimeError('Input must be a float tensor')

    # [TODO] Fix reduce-like shape inference. We now assume all inputs are 4-D.
    output_shape = [0, 0, 0, 0]
    for i in range(4):
        input_dims = [variable.type.shape[i] for variable in operator.inputs]
        if 'None' in input_dims:
            output_shape[i] = 'None'
        else:
            output_shape[i] = max(input_dims)

    operator.outputs[0].type.shape = output_shape


def calculate_upsample_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')
    scales = operator.raw_operator.upsample.scalingFactor

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    output_shape[2] *= scales[0]
    output_shape[3] *= scales[1]

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


def calculate_split_output_shapes(operator):
    if len(operator.inputs) != 1:
        raise RuntimeError('Split has only one input')

    if len(operator.inputs) < 1:
        raise RuntimeError('Split should create at least one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    divided = output_shape[1] / operator.raw_operator.split.nOutputs
    if divided != int(divided):
        raise RuntimeError('Variable dimension along C-axis must be divisible by partition number')

    output_shape[1] = int(divided)

    operator.outputs[0].type.shape = output_shape


def calculate_slice_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Slice has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.slice

    from coremltools.proto.NeuralNetwork_pb2 import SliceLayerParams as Params
    axis_map = {Params.CHANNEL_AXIS: 1, Params.HEIGHT_AXIS: 2, Params.WIDTH_AXIS: 3}

    if params.startIndex >= 0:
        output_shape[axis_map[Params.CHANNEL_AXIS]] = params.endIndex - params.startIndex
    else:
        output_shape[axis_map[Params.CHANNEL_AXIS]] += 1 + params.endIndex - params.startIndex

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


def calculate_sequence_repeat_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Sequence Repeqt has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    if output_shape[0] != None:
        output_shape[0] *= operator.raw_operator.sequenceRepeat.nRepetitions

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


def calculate_reorganizeData_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reorganize Data has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.reorganizeData

    from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params

    if params.mode == Params.DEPTH_TO_SPACE:
        if output_shape[1] % (params.blockSize * params.blockSize) != 0:
            raise RuntimeError('Channel number must be divisible by the square of block size')

        output_shape = [output_shape[0], output_shape[1] / params.blockSize / params.blockSize,
                        output_shape[2] * params.blockSize, output_shape[3] * params.blockSize ]
    elif params.mode == Params.SPACE_TO_DEPTH:
        if output_shape[2] % params.blockSize != 0 or output_shape[3] % params.blockSize != 0:
            raise RuntimeError('Height and weight must be divisible by block size')

        output_shape = [output_shape[0], output_shape[1] * params.blockSize * params.blockSize,
                        output_shape[2] / params.blockSize, output_shape[3] / params.blockSize ]
    else:
        raise ValueError('Unsupport reorganization mode {0}'.format(params.mode))

    operator.outputs[0].type = FloatTensorType([int(i) if i != 'None' else 'None' for i in output_shape],
                                               doc_string=operator.outputs[0].type.doc_string)


def calculate_reduce_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reduce has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    params = operator.raw_operator.reduce

    from coremltools.proto.NeuralNetwork_pb2 import ReduceLayerParams as Params
    # Adjust C-axis
    if params.axis in [Params.CHW, Params.C]:
        output_shape[1] = 1
    # Adjust H-axis
    if params.axis in [Params.CHW, Params.HW, Params.H]:
        output_shape[2] = 1
    # Adjust W-axis
    if params.axis in [Params.CHW, Params.HW, Params.W]:
        output_shape[3] = 1

    operator.outputs[0].type.shape = output_shape


def calculate_load_constant_output_shapes(operator):
    if len(operator.inputs) != 0:
        raise RuntimeError('Load Constant operator has no input')
    if len(operator.outputs) != 1:
        raise RuntimeError('Load Constant operator has only one output')
    output = operator.outputs[0]

    # CoreML's constant is always 3-D tensor, so we assume its shape is [C, H, W].
    const_shape = operator.raw_operator.loadConstant.shape
    # We convert [C, H, W] to [1, C, H, W] because our parsing code use [N, C, H, W]
    const_shape = [1] + [int(d) for d in const_shape]
    if output.type is None:
        # Use default type
        output.type = FloatTensorType(const_shape, doc_string=output.type.doc_string)
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

    # Assume that inputs are [N, C]- or [N, C, 1, 1]-tensors
    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    output_shape[1] = 1
    operator.outputs[0].type.shape = output_shape


type_calculator_table = {'activation': calculate_identical_float_tensor_shapes,
                         'innerProduct': calculate_inner_product_output_shapes,
                         'identity': calculate_identity_output_shapes,
                         'softmax': calculate_identical_float_tensor_shapes,
                         'convolution': calculate_convolution_output_shapes,
                         'pooling': calculate_pooling_output_shapes,
                         'scalerPreprocessor': calculate_identical_float_tensor_shapes,
                         'flatten': calculate_flatten_output_shapes,
                         'permute': calculate_permute_output_shapes,
                         'imageToFloatTensor': calculate_identical_float_tensor_shapes,
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
                         'unary': calculate_identical_float_tensor_shapes,
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
                         'normalizer': calculate_identical_float_tensor_shapes
                         }
