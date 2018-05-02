import unittest
import cntk
import coremltools
import onnxmltools
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, \
    Dot, Embedding, BatchNormalization, GRU, Activation, PReLU, LeakyReLU, ThresholdedReLU, Maximum, \
    Add, Average, Multiply, Concatenate, UpSampling2D, Flatten, RepeatVector
from keras.initializers import RandomUniform


def _create_keras_and_cntk_2d_inputs(N, C):
    np.random.seed(0)
    x_keras = np.random.rand(N, C)
    x_cntk = [np.ascontiguousarray(np.squeeze(_)) for _ in np.split(x_keras, N)]
    return x_keras, x_cntk


def _create_keras_and_cntk_4d_inputs(N, C, H, W):
    np.random.seed(0)
    x_keras = np.random.rand(N, C, H, W)
    x_cntk = [np.ascontiguousarray(np.squeeze(_, axis=0)) for _ in np.split(x_keras, N)]
    return x_keras, x_cntk


def _create_keras_and_cntk_2d_input_pair(N, C):
    np.random.seed(0)
    x_keras1 = np.random.rand(N, C)
    x_keras2 = np.random.rand(N, C)
    x_cntk1 = [np.ascontiguousarray(np.squeeze(_)) for _ in np.split(x_keras1, N)]
    x_cntk2 = [np.ascontiguousarray(np.squeeze(_)) for _ in np.split(x_keras2, N)]
    return x_keras1, x_keras2, x_cntk1, x_cntk2


def _create_keras_and_cntk_4d_input_pair(N, C, H, W):
    np.random.seed(0)
    x_keras1 = np.random.rand(N, C, H, W)
    x_keras2 = np.random.rand(N, C, H, W)
    x_cntk1 = [np.ascontiguousarray(np.squeeze(_, axis=0)) for _ in np.split(x_keras1, N)]
    x_cntk2 = [np.ascontiguousarray(np.squeeze(_, axis=0)) for _ in np.split(x_keras2, N)]
    return x_keras1, x_keras2, x_cntk1, x_cntk2


class TestKeras2CoreML2ONNXWithCNTK(unittest.TestCase):

    def _test_one_to_one_operator_core(self, keras_model, x_keras, x_cntk):
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

        y_keras = keras_model.predict(x_keras)
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def _test_one_to_one_operator_core_channels_last(self, keras_model, x_keras, x_cntk):
        '''
        Keras computation path:
            [N, C, H, W] ---> numpy transpose ---> [N, H, W, C] ---> keras convolution --->
            [N, H, W, C] ---> numpy transpose ---> [N, C, H, W]

        ONNX computation path:
            [N, C, H, W] ---> ONNX convolution ---> [N, C, H, W]

        The reason for having extra transpose's in the Keras path is that CoreMLTools doesn't not handle channels_last
        flag properly. Precisely, oreMLTools always converts Conv2D under channels_first mode.
        '''
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        y_keras = np.transpose(keras_model.predict(np.transpose(x_keras, [0, 2, 3, 1])), [0, 3, 1, 2])

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def _test_two_to_one_operator_core(self, keras_model, x_keras1, x_keras2, x_cntk1, x_cntk2):
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

        y_keras = keras_model.predict([x_keras1, x_keras2])
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk1, cntk_model.arguments[1]: x_cntk2})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def _test_two_to_one_operator_core_channels_last(self, keras_model, x_keras1, x_keras2, x_cntk1, x_cntk2):
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

        y_keras = np.transpose(keras_model.predict(
            [np.transpose(x_keras1, [0, 2, 3, 1]), np.transpose(x_keras2, [0, 2, 3, 1])]), [0, 3, 1, 2])
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk1, cntk_model.arguments[1]: x_cntk2})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def test_dense(self):
        x_keras, x_cntk = _create_keras_and_cntk_2d_inputs(2, 3)

        np.random.seed(0)
        model = Sequential()
        model.add(Dense(2, input_dim=3))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core(model, x_keras, x_cntk)

    def test_conv_2d(self):
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(1, 2, 4, 3)
        np.random.seed(0)
        model = Sequential()
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(4, 3, 2),
                         data_format='channels_last'))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_max_pooling_2d(self):
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(1, 2, 4, 3)
        np.random.seed(0)
        model = Sequential()
        model.add(MaxPooling2D(2, input_shape=(4, 3, 2), data_format='channels_last'))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_average_pooling_2d(self):
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(1, 2, 4, 3)
        np.random.seed(0)
        model = Sequential()
        model.add(AveragePooling2D(2, input_shape=(4, 3, 2), data_format='channels_last'))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    @unittest.skip('Skip because CNTK is not able to evaluate this model')
    def test_convolution_transpose_2d(self):
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 2, 1, 1)
        np.random.seed(0)
        model = Sequential()
        model.add(Conv2DTranspose(2, (2, 1), input_shape=(1, 1, 2), data_format='channels_last'))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_maximum(self):
        N = 2
        C = 2
        H = 1
        W = 1
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_4d_input_pair(N, C, H, W)
        input1 = Input(shape=(H, W, C))
        input2 = Input(shape=(H, W, C))
        result = Maximum()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core_channels_last(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_maximum_2d(self):
        N = 2
        C = 2
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_2d_input_pair(N, C)
        input1 = Input(shape=(C,))
        input2 = Input(shape=(C,))
        result = Maximum()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_dot(self):
        N = 2
        C = 2
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_2d_input_pair(N, C)
        input1 = Input(shape=(C,))
        input2 = Input(shape=(C,))
        result = Dot(axes=-1)([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_add(self):
        N = 2
        C = 2
        H = 1
        W = 1
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_4d_input_pair(N, C, H, W)
        input1 = Input(shape=(H, W, C))
        input2 = Input(shape=(H, W, C))
        result = Add()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core_channels_last(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_concatenate(self):
        N = 2
        C = 2
        H = 1
        W = 1
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_4d_input_pair(N, C, H, W)
        input1 = Input(shape=(H, W, C))
        input2 = Input(shape=(H, W, C))
        result = Concatenate()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core_channels_last(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_multiply(self):
        N = 2
        C = 2
        H = 1
        W = 1
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_4d_input_pair(N, C, H, W)
        input1 = Input(shape=(H, W, C))
        input2 = Input(shape=(H, W, C))
        result = Multiply()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core_channels_last(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    def test_average(self):
        N = 2
        C = 2
        H = 1
        W = 1
        x_keras1, x_keras2, x_cntk1, x_cntk2 = _create_keras_and_cntk_4d_input_pair(N, C, H, W)
        input1 = Input(shape=(H, W, C))
        input2 = Input(shape=(H, W, C))
        result = Average()([input1, input2])
        model = Model(inputs=[input1, input2], output=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_two_to_one_operator_core_channels_last(model, x_keras1, x_keras2, x_cntk1, x_cntk2)

    @unittest.skip('CNTK does not support integer tensor as its input')
    def test_embedding(self):
        x_keras = np.array([[1], [2], [0]])
        x_cntk = [np.array([1]), np.array([2]), np.array([0])]
        keras_model = Sequential()
        keras_model.add(Embedding(3, 2, input_length=1))
        keras_model.compile(optimizer='adagrad', loss='mse')

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

        y_keras = keras_model.predict(x_keras)
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def test_batch_normalization(self):
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 2, 3, 4)
        model = Sequential()
        model.add(BatchNormalization(beta_initializer='random_uniform', gamma_initializer='random_uniform',
                                     moving_mean_initializer='random_uniform',
                                     moving_variance_initializer=RandomUniform(minval=0.1, maxval=0.5),
                                     input_shape=(3, 4, 2)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    @unittest.skip('CNTK does not support reshape under this configuration')
    def test_gru(self):
        np.random.seed(0)
        N = 1
        T = 2
        C = 3
        D = 2
        input = Input(shape=(T, C))
        rnn = GRU(D)
        result = rnn(input)
        keras_model = Model([input], [result])
        keras_model.compile(optimizer='adagrad', loss='mse')

        x_keras = np.random.rand(N, T, C)
        x_cntk = [np.squeeze(_, axis=0) for _ in np.split(np.random.rand(T, C), T)]

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

        y_keras = keras_model.predict(x_keras)
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk})

        self.assertTrue(np.allclose(y_keras, y_cntk))

    def test_activation_tanh(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('tanh', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_relu(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('relu', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_elu(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('elu', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    @unittest.skip('CNTK does not support SELU')
    def test_activation_selu(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('selu', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_softplus(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('softplus', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_softsign(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('softsign', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_sigmoid(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('sigmoid', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_hard_sigmoid(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 4, 5)
        model = Sequential()
        model.add(Activation('hard_sigmoid', input_shape=(4, 5, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_activation_leaky_relu(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_2d_inputs(2, 3)
        model = Sequential()
        model.add(LeakyReLU(input_shape=(3,)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core(model, x_keras, x_cntk)

    @unittest.skip('CNTK does not support Upsample operator')
    def test_upsample(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 1, 2)

        model = Sequential()
        model.add(UpSampling2D(input_shape=(1, 2, 3)))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x_keras, x_cntk)

    def test_flatten(self):
        np.random.seed(0)
        x_keras, x_cntk = _create_keras_and_cntk_4d_inputs(2, 3, 1, 2)

        keras_model = Sequential()
        keras_model.add(Flatten(input_shape=(1, 2, 3)))
        keras_model.add(Dense(2))
        keras_model.compile(optimizer='adagrad', loss='mse')

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        y_keras = keras_model.predict(np.transpose(x_keras, [0, 2, 3, 1]))

        temporary_onnx_model_file_name = onnx_model.graph.name + '_temp.onnx'
        onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
        cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)
        y_cntk = cntk_model.eval({cntk_model.arguments[0]: x_cntk})

        self.assertTrue(np.allclose(y_keras, y_cntk))
