import unittest
import numpy
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes
from onnxmltools.convert.coreml.CoremlConvertContext import CoremlConvertContext as ConvertContext
from onnxmltools.convert.coreml.NeuralNetwork.activation import ActivationConverter
from onnxmltools.convert.coreml.NeuralNetwork.add import AddLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.average import AverageLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.batchnorm import BatchnormLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.bias import BiasLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.biDirectionalLSTM import BiDirectionalLSTMLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.concat import ConcatLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.convolution import ConvolutionLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.crop import CropLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.dot import DotProductLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.embedding import EmbeddingLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.flatten import FlattenLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.fullyconnected import InnerProductLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.gru import GRULayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.l2normalize import L2NormalizeLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.loadconstant import LoadConstantLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.lrn import LRNLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.max import MaxLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.meanvariance import MeanVarianceNormalizeLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.min import MinLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.multiply import MultiplyLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.padding import PaddingLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.permute import PermuteLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.pooling import PoolingLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.reduce import ReduceLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.reorganizeData import ReorganizeDataLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.reshape import ReshapeLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.scale import ScaleLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.sequencerepeat import SequenceRepeatLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.simpleRecurrent import SimpleRecurrentLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.slice import SliceLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.softmax import SoftmaxLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.split import SplitLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.unaryfunction import UnaryFunctionLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.uniDirectionalLSTM import UniDirectionalLSTMLayerConverter
from onnxmltools.convert.coreml.NeuralNetwork.upsample import UpsampleLayerConverter


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
        context = ConvertContext()
        node = InnerProductLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_unary_function_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_unary(name='Unary1', input_name='input', output_name='mid1', mode='abs')
        builder.add_unary(name='Unary3', input_name='mid1', output_name='mid2', mode='sqrt')
        builder.add_unary(name='Unary2', input_name='mid2', output_name='mid3', mode='rsqrt')
        builder.add_unary(name='Unary3', input_name='mid3', output_name='mid4', mode='inverse')
        builder.add_unary(name='Unary4', input_name='mid4', output_name='mid5', mode='power', alpha=2)
        builder.add_unary(name='Unary5', input_name='mid5', output_name='mid6', mode='exp')
        builder.add_unary(name='Unary6', input_name='mid6', output_name='mid7', mode='log')
        builder.add_unary(name='Unary7', input_name='mid7', output_name='output', mode='threshold')
        context = ConvertContext()
        for layer in builder.spec.neuralNetwork.layers:
            for node in UnaryFunctionLayerConverter.convert(context, layer, ['input'], ['output']):
                self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = ConvolutionLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_pooling_converter(self):
        input_dim = (1, 1, 4, 2)
        output_dim = (1, 1, 4, 2)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_pooling(name='Pool', height=2, width=2, stride_height=1, stride_width=1, layer_type='MAX',
                            padding_type='SAME', input_name='input', output_name='output')
        context = ConvertContext()
        node = PoolingLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_activation_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_activation(name='Activation', non_linearity='RELU', input_name='input', output_name='output')
        context = ConvertContext()
        node = ActivationConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = EmbeddingLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = BatchnormLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_mean_variance_normalize_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_mvn(name='MVN', input_name='input', output_name='output', epsilon=0)
        context = ConvertContext()
        node = MeanVarianceNormalizeLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'],
                                                           ['output'])
        self.assertTrue(node is not None)

    def test_l2_normalize_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_l2_normalize(name='L2', input_name='input', output_name='output')
        context = ConvertContext()
        node = L2NormalizeLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_softmax_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_softmax(name='Softmax', input_name='input', output_name='output')
        context = ConvertContext()
        node = SoftmaxLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_lrn_converter(self):
        input_dim = (3,)
        output_dim = (3,)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_lrn(name='LRN', input_name='input', output_name='output', alpha=0.5, beta=2, k=1, local_size=2)
        context = ConvertContext()
        node = LRNLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

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
                         input_names='input', output_name='output')
        context = ConvertContext()
        node = CropLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_padding_converter(self):
        input_dim = (1, 3, 4)
        output_dim = (1, 5, 6)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_padding(name='Pad', left=2, right=0, top=2, bottom=0, input_name='input', output_name='output',
                            padding_type='constant')
        context = ConvertContext()
        node = PaddingLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_upsample_converter(self):
        input_dim = (1, 1, 1)
        output_dim = (1, 2, 2)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        builder.add_upsample(name='Upsample', scaling_factor_h=2, scaling_factor_w=2, input_name='input',
                             output_name='output')
        context = ConvertContext()
        node = UpsampleLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_add_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Add', input_names=['input1', 'input2'], output_name='output', mode='ADD')
        context = ConvertContext()
        node = AddLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input1', 'input2'],
                                         ['output'])
        self.assertTrue(node is not None)

    def test_multiply_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Mul', input_names=['input1', 'input2'], output_name='output', mode='MULTIPLY')
        context = ConvertContext()
        node = MultiplyLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_average_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='MEAN', input_names=['input1', 'input2'], output_name='output', mode='AVE')
        context = ConvertContext()
        node = AverageLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = ScaleLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_bias_converter(self):
        input_dim = (2, 1, 1)
        output_dim = (2, 1, 1)
        input = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(input, output)
        bias = numpy.ndarray(shape=(2,))
        bias[:] = [1, 2]
        builder.add_bias(name='Bias', b=bias, input_name='input', output_name='output', shape_bias=[2])
        context = ConvertContext()
        node = BiasLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_max_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Max', input_names=['input1', 'input2'], output_name='output', mode='MAX')
        context = ConvertContext()
        node = MaxLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_min_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1, 2, 2)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Min', input_names=['input1', 'input2'], output_name='output', mode='MIN')
        context = ConvertContext()
        node = MinLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_dot_product_converter(self):
        input_dim = (3,)
        output_dim = (1,)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_elementwise(name='Dot', input_names=['input1', 'input2'], output_name='output', mode='DOT')
        context = ConvertContext()
        node = DotProductLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_reduce_converter(self):
        input_dim = (1, 2, 2)
        output_dim = (1,)
        inputs = [('input', datatypes.Array(*input_dim))]
        output = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, output)
        builder.add_reduce(name='Reduce', input_name='input', output_name='output', axis='CHW', mode='sum')
        context = ConvertContext()
        node = ReduceLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_load_constant_converter(self):
        value = numpy.ndarray(shape=(1, 1, 2))
        value[:] = [[[-95, 95]]]
        shape = value.shape
        inputs = [('const', datatypes.Array(*shape))]
        outputs = [('const', datatypes.Array(*shape))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_load_constant(name='LoadConstant', output_name='const', constant_value=value, shape=shape)
        context = ConvertContext()
        node = LoadConstantLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_reshape_converter(self):
        input_dim = (1, 1, 2)
        output_dim = (1, 2, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_reshape(name='Reshape', input_name='input', output_name='output', target_shape=output_dim, mode=1)
        context = ConvertContext()
        node = ReshapeLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_flatten_converter(self):
        input_dim = (1, 2, 3)
        output_dim = (6, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_flatten(name='Flatten', input_name='input', output_name='output', mode=1)
        context = ConvertContext()
        node = FlattenLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_permute_converter(self):
        input_dim = (4, 1, 2, 3)
        output_dim = (4, 3, 1, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_permute(name='Permute', input_name='input', output_name='output', dim=(0, 2, 3, 1))
        context = ConvertContext()
        node = PermuteLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_concat_converter(self):
        input_dim = (5, 1, 1)
        output_dim = (10, 1, 1)
        inputs = [('input1', datatypes.Array(*input_dim)), ('input2', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_elementwise(name='Concate', input_names=['input1', 'input2'], output_name='output', mode='CONCAT')
        context = ConvertContext()
        node = ConcatLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_split_converter(self):
        input_dim = (8, 1, 1)
        output_dim = (4, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output1', datatypes.Array(*output_dim)), ('output2', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_split(name='Split', input_name='input', output_names=['output1', 'output2'])
        context = ConvertContext()
        node = SplitLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

    def test_sequence_repeat_converter(self):
        input_dim = (3, 1, 1)
        output_dim = (9, 1, 1)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_sequence_repeat(name='Repeat', input_name='input', output_name='output', nrep=3)
        context = ConvertContext()
        node = SequenceRepeatLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'],
                                                    ['output'])
        self.assertTrue(node is not None)

    def test_reorganize_data_converter(self):
        block_size = 2
        input_dim = (3, 4 * block_size, 2 * block_size)
        output_dim = (3 * block_size * block_size, 4, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_reorganize_data(name='Reorg', input_name='input', output_name='output', mode='SPACE_TO_DEPTH',
                                    block_size=2)
        context = ConvertContext()
        node = ReorganizeDataLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'],
                                                    ['output'])
        self.assertTrue(node is not None)

    def test_slice_converter(self):
        input_dim = (1, 4, 2)
        output_dim = (1, 2, 2)
        inputs = [('input', datatypes.Array(*input_dim))]
        outputs = [('output', datatypes.Array(*output_dim))]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_slice(name='Slice', input_name='input', output_name='output', axis='height', start_index=0,
                          end_index=-1, stride=1)
        context = ConvertContext()
        node = SliceLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'], ['output'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = GRULayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input', 'h_init'],
                                         ['output', 'h'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = SimpleRecurrentLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0], ['input'],
                                                     ['output'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = UniDirectionalLSTMLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0],
                                                        ['input', 'h_init', 'c_init'], ['output', 'h', 'c'])
        self.assertTrue(node is not None)

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
        context = ConvertContext()
        node = BiDirectionalLSTMLayerConverter.convert(context, builder.spec.neuralNetwork.layers[0],
                                                       ['input', 'h_init', 'c_init', 'h_back_init', 'c_back_init'],
                                                       ['output', 'h', 'c', 'h_back', 'c_back'])
        self.assertTrue(node is not None)
