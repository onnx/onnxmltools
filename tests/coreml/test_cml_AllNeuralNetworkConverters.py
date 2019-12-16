import unittest
import numpy
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes
from coremltools.proto.FeatureTypes_pb2 import ImageFeatureType
from distutils.version import StrictVersion
from onnxmltools import convert_coreml
from onnxmltools.proto import onnx

class TestNeuralNetworkLayerConverter(unittest.TestCase):

    def test_inner_product_converter(self):
        input_dim = (3,)
        output_dim = (2,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        weights = numpy.zeros(shape=(3, 2))
        weights[:] = [[1, 2], [3, 4], [5, 6]]
        bias = numpy.zeros(shape=(2))
        bias[:] = [-100, 100]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_inner_product(name='FC', W=weights, b=bias, input_channels=3, output_channels=2, has_bias=True,
                                  input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_unary_function_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_unary(name='Unary1', input_name='input', output_name='mid1', mode='abs')
        builder.add_unary(name='Unary2', input_name='mid1', output_name='mid2', mode='sqrt')
        builder.add_unary(name='Unary3', input_name='mid2', output_name='mid3', mode='rsqrt')
        builder.add_unary(name='Unary4', input_name='mid3', output_name='mid4', mode='inverse')
        builder.add_unary(name='Unary5', input_name='mid4', output_name='mid5', mode='power', alpha=2)
        builder.add_unary(name='Unary6', input_name='mid5', output_name='mid6', mode='exp')
        builder.add_unary(name='Unary7', input_name='mid6', output_name='mid7', mode='log')
        builder.add_unary(name='Unary8', input_name='mid7', output_name='output', mode='threshold')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_convolution_converter(self):
        input_dim = (1, 1, 4, 2)
        output_dim = (1, 1, 4, 2)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        weights = numpy.zeros(shape=(1, 1, 2, 2))
        weights[:] = [[1, 1], [-1, -1]]
        bias = numpy.zeros(shape=(1,))
        bias[:] = 100
        builder.add_convolution(name='Conv', kernel_channels=1, output_channels=1, height=2, width=2, stride_height=1,
                                stride_width=1, border_mode='same', groups=1,
                                W=weights, b=bias, has_bias=True, input_name='input', output_name='output',
                                is_deconv=True, output_shape=(1, 1, 4, 2))
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_pooling_converter(self):
        input_dim = (1, 1, 4, 2)
        output_dim = (1, 1, 4, 2)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_pooling(name='Pool', height=2, width=2, stride_height=1, stride_width=1, layer_type='MAX',
                            padding_type='SAME', input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_activation_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_activation(name='Activation', non_linearity='RELU', input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_embedding_converter(self):
        input_dim = (1, 1, 1, 1)
        output_dim = (1, 2, 1, 1)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        weights = numpy.zeros(shape=(2))
        weights[:] = [-1, 1]
        bias = numpy.zeros(shape=(2))
        bias[:] = [-100, 100]
        builder.add_embedding(name='Embed', input_dim=1, W=weights, b=bias, output_channels=2, has_bias=True,
                              input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_batchnorm_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        gamma = numpy.ndarray(shape=(3,))
        gamma[:] = [-1, 0, 1]
        beta = numpy.ndarray(shape=(3,))
        beta[:] = [10, 20, 30]
        mean = numpy.ndarray(shape=(3,))
        mean[:] = [0, 0, 0]
        variance = numpy.ndarray(shape=(3,))
        variance[:] = [1, 1, 1]
        builder.add_batchnorm(name='BatchNormalize', channels=3, gamma=gamma, beta=beta, mean=mean, variance=variance,
                              input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_mean_variance_normalize_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_mvn(name='MVN', input_name='input', output_name='output', epsilon=0)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_l2_normalize_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_l2_normalize(name='L2', input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_softmax_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_softmax(name='Softmax', input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_lrn_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_lrn(name='LRN', input_name='input', output_name='output', alpha=0.5, beta=2, k=1, local_size=2)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_crop_converter(self):
        top_crop = 1
        bottom_crop = 1
        left_crop = 1
        right_crop = 1
        input_dim = (8, 6, 4)
        output_dim = (8, 6 - top_crop - bottom_crop, 4 - left_crop - right_crop)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_crop(name='Crop', left=left_crop, right=right_crop, top=top_crop, bottom=bottom_crop, offset=[0, 0],
                         input_names=['input'], output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_padding_converter(self):
        input_dim = (1, 3, 4)
        output_dim = (1, 5, 6)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_padding(name='Pad', left=2, right=0, top=2, bottom=0, input_name='input', output_name='output',
                            padding_type='constant')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_upsample_converter(self):
        input_dim = (1, 1, 1)
        output_dim = (1, 2, 2)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_upsample(name='Upsample', scaling_factor_h=2, scaling_factor_w=2, input_name='input',
                             output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_add_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Add', input_names=['input1', 'input2'], output_name='output', mode='ADD')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_multiply_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Mul', input_names=['input1', 'input2'], output_name='output', mode='MULTIPLY')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_average_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='MEAN', input_names=['input1', 'input2'], output_name='output', mode='AVE')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_scale_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        scale = numpy.ndarray(shape=(1,))
        scale[:] = 10
        bias = numpy.ndarray(shape=(1,))
        bias[:] = -100
        builder.add_scale(name='ImageScaler', W=scale, b=bias, has_bias=True, input_name='input', output_name='output')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_bias_converter(self):
        input_dim = (2, 1, 1)
        output_dim = (2, 1, 1)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        bias = numpy.ndarray(shape=(2,))
        bias[:] = [1, 2]
        builder.add_bias(name='Bias', b=bias, input_name='input', output_name='output', shape_bias=[2])
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_max_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Max', input_names=['input1', 'input2'], output_name='output', mode='MAX')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_min_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Min', input_names=['input1', 'input2'], output_name='output', mode='MIN')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_dot_product_converter(self):
        input_dim = (3,)
        output_dim = (1,)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Dot', input_names=['input1', 'input2'], output_name='output', mode='DOT')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_reduce_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1,)
        inputs = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_reduce(name='Reduce', input_name='input', output_name='output', axis='CHW', mode='sum')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_load_constant_converter(self):
        value = numpy.ndarray(shape=(1, 1, 2))
        value[:] = [[[-95, 95]]]
        shape = value.shape
        input_dim = (1, 2, 3, 4)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('const', datatypes.Array(*shape)), ('output', datatypes.Array(*input_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_load_constant(name='LoadConstant', output_name='const', constant_value=value, shape=shape)
        builder.add_permute(name='Permute', input_name='input', output_name='output', dim=(0, 1, 2, 3))
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_reshape_converter(self):
        input_dim = (1, 1, 2)
        output_dim = (1, 2, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_reshape(name='Reshape', input_name='input', output_name='output', target_shape=output_dim, mode=1)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_flatten_converter(self):
        input_dim = (1, 2, 3)
        output_dim = (6, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_flatten(name='Flatten', input_name='input', output_name='output', mode=1)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_permute_converter(self):
        input_dim = (4, 1, 2, 3)
        output_dim = (4, 3, 1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_permute(name='Permute', input_name='input', output_name='output', dim=(0, 2, 3, 1))
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_concat_converter(self):
        input_dim = (5, 1, 1)
        output_dim = (10, 1, 1)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_elementwise(name='Concate', input_names=['input1', 'input2'], output_name='output', mode='CONCAT')
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_split_converter(self):
        input_dim = (8, 1, 1)
        output_dim = (4, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output1', datatypes.Array(*output_dim)), ('output2', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_split(name='Split', input_name='input', output_names=['output1', 'output2'])
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_sequence_repeat_converter(self):
        input_dim = (3, 1, 1)
        output_dim = (9, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_sequence_repeat(name='Repeat', input_name='input', output_name='output', nrep=3)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_reorganize_data_converter(self):
        block_size = 2
        input_dim = (3, 4 * block_size, 2 * block_size)
        output_dim = (3 * block_size * block_size, 4, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_reorganize_data(name='Reorg', input_name='input', output_name='output', mode='SPACE_TO_DEPTH',
                                    block_size=2)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_slice_converter(self):
        input_dim = (1, 4, 2)
        output_dim = (1, 2, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_slice(name='Slice', input_name='input', output_name='output', axis='height', start_index=0,
                          end_index=-1, stride=1)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_gru_converter(self):
        input_dim = (1, 8)
        output_dim = (1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        W_h = [numpy.random.rand(2, 2), numpy.random.rand(2, 2), numpy.random.rand(2, 2)]
        W_x = [numpy.random.rand(2, 8), numpy.random.rand(2, 8), numpy.random.rand(2, 8)]
        b = [numpy.random.rand(2, 1), numpy.random.rand(2, 1), numpy.random.rand(2, 1)]
        builder.add_gru(name='GRU', W_h=W_h, W_x=W_x, b=b, hidden_size=2, input_size=8, input_names=['input'],
                        output_names=['output'], activation='TANH', inner_activation='SIGMOID_HARD', output_all=False,
                        reverse_input=False)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_simple_recurrent_converter(self):
        input_dim = (1, 8)
        output_dim = (1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        W_h = numpy.random.rand(2, 2)
        W_x = numpy.random.rand(2, 8)
        b = numpy.random.rand(2, 1)
        builder.add_simple_rnn(name='RNN', W_h=W_h, W_x=W_x, b=b, hidden_size=2, input_size=8,
                               input_names=['input', 'h_init'], output_names=['output', 'h'], activation='TANH',
                               output_all=False, reverse_input=False)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_unidirectional_lstm_converter(self):
        input_dim = (1, 8)
        output_dim = (1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        W_h = [numpy.random.rand(2, 2), numpy.random.rand(2, 2), numpy.random.rand(2, 2), numpy.random.rand(2, 2)]
        W_x = [numpy.random.rand(2, 8), numpy.random.rand(2, 8), numpy.random.rand(2, 8), numpy.random.rand(2, 8)]
        b = [numpy.random.rand(2, 1), numpy.random.rand(2, 1), numpy.random.rand(2, 1), numpy.random.rand(2, 1)]
        p = [numpy.zeros(shape=(2, 1)), numpy.zeros(shape=(2, 1)), numpy.zeros(shape=(2, 1))]
        builder.add_unilstm(name='LSTM', W_h=W_h, W_x=W_x, b=b, hidden_size=2, input_size=8, input_names=['input'],
                            output_names=['output'], inner_activation='SIGMOID', cell_state_update_activation='TANH',
                            output_activation='TANH', peep=p, output_all=False, forget_bias=False,
                            coupled_input_forget_gate=False, cell_clip_threshold=10000, reverse_input=False)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_bidirectional_lstm_converter(self):
        input_dim = (1, 8)
        output_dim = (1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        W_h = [numpy.random.rand(2, 2), numpy.random.rand(2, 2), numpy.random.rand(2, 2), numpy.random.rand(2, 2)]
        W_x = [numpy.random.rand(2, 8), numpy.random.rand(2, 8), numpy.random.rand(2, 8), numpy.random.rand(2, 8)]
        b = [numpy.random.rand(2, 1), numpy.random.rand(2, 1), numpy.random.rand(2, 1), numpy.random.rand(2, 1)]
        p = [numpy.zeros(shape=(2, 1)), numpy.zeros(shape=(2, 1)), numpy.zeros(shape=(2, 1))]
        builder.add_bidirlstm(name='LSTM', W_h=W_h, W_x=W_x, W_h_back=W_h, b=b, W_x_back=W_x, b_back=b, hidden_size=2,
                              input_size=8, input_names=['input'], output_names=['output'], inner_activation='SIGMOID',
                              cell_state_update_activation='TANH', output_activation='TANH', peep=p, peep_back=p,
                              output_all=False, forget_bias=False, coupled_input_forget_gate=False,
                              cell_clip_threshold=10000)
        model_onnx = convert_coreml(builder.spec)
        self.assertTrue(model_onnx is not None)

    def test_image_input_type_converter(self):
        dim = (3, 15, 25)
        inputs = [('input', datatypes.Array(*dim))]
        outputs = [('output', datatypes.Array(*dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_elementwise(name='Identity', input_names=['input'],
                                output_name='output', mode='ADD', alpha=0.0)
        spec = builder.spec
        input = spec.description.input[0]
        input.type.imageType.height = dim[1]
        input.type.imageType.width = dim[2]
        for coreml_colorspace, onnx_colorspace in (('RGB', 'Rgb8'), ('BGR', 'Bgr8'), ('GRAYSCALE', 'Gray8')):
            input.type.imageType.colorSpace = ImageFeatureType.ColorSpace.Value(coreml_colorspace)
            model_onnx = convert_coreml(spec)
            dims = [(d.dim_param or d.dim_value) for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
            self.assertEqual(dims, ['None', 1 if onnx_colorspace == 'Gray8' else 3, 15, 25])

            if StrictVersion(onnx.__version__) >= StrictVersion('1.2.1'):
                metadata = {prop.key: prop.value for prop in model_onnx.metadata_props}
                self.assertEqual(metadata, { 'Image.BitmapPixelFormat': onnx_colorspace })
                self.assertEqual(model_onnx.graph.input[0].type.denotation, 'IMAGE')
                channel_denotations = [d.denotation for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
                self.assertEqual(channel_denotations, ['DATA_BATCH', 'DATA_CHANNEL', 'DATA_FEATURE', 'DATA_FEATURE'])
