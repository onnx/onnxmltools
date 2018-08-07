# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

import coremltools
import numpy as np
import unittest
import onnx
import onnxmltools
from distutils.version import StrictVersion
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, \
    Dot, Embedding, BatchNormalization, GRU, Activation, PReLU, LeakyReLU, ThresholdedReLU, Maximum, \
    Add, Average, Multiply, Concatenate, UpSampling2D, Flatten, RepeatVector, Reshape, Dropout
from keras.initializers import RandomUniform

np.random.seed(0)

def _find_backend():
    try:
        import cntk
        return 'cntk'
    except:
        pass
    try:
        import caffe2
        return 'caffe2'
    except:
        pass
    return None


def _evaluate(onnx_model, inputs):
    runtime_name = _find_backend()
    if runtime_name == 'cntk':
        return _evaluate_cntk(onnx_model, inputs)
    elif runtime_name == 'caffe2':
        return _evaluate_caffe2(onnx_model, inputs)
    else:
        raise RuntimeError('No runtime found. Need either CNTK or Caffe2')


def _evaluate_caffe2(onnx_model, inputs):
    from caffe2.python.onnx.backend import run_model

    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = run_model(onnx_model, inputs)

    adjusted_outputs = dict()
    for output in onnx_model.graph.output:
        adjusted_outputs[output.name] = outputs[output.name]

    return adjusted_outputs[onnx_model.graph.output[0].name]


def _evaluate_cntk(onnx_model, inputs):
    import cntk
    if not isinstance(inputs, list):
        inputs = [inputs]

    adjusted_inputs = dict()

    for i, x in enumerate(inputs):
        onnx_name = onnx_model.graph.input[i].name
        adjusted_inputs[onnx_name] = [np.ascontiguousarray(np.squeeze(_, axis=0)) for _ in np.split(x, x.shape[0])]

    temporary_onnx_model_file_name = 'temp_' + onnx_model.graph.name + '.onnx'
    onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
    cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

    return cntk_model.eval(adjusted_inputs)


def _create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor')


class TestKeras2CoreML2ONNX(unittest.TestCase):

    def _test_one_to_one_operator_core(self, keras_model, x):
        # Verify Keras-to-CoreML-to-ONNX path
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        y_reference = keras_model.predict(x)
        y_produced = _evaluate(onnx_model, x)

        self.assertTrue(np.allclose(y_reference, y_produced))

        # Verify Keras-to-ONNX path
        onnx_model = onnxmltools.convert_keras(keras_model)
        y_produced = _evaluate(onnx_model, x)

        self.assertTrue(np.allclose(y_reference, y_produced))

    def _test_one_to_one_operator_core_keras(self, keras_model, x):
        y_reference = keras_model.predict(x)

        onnx_model = onnxmltools.convert_keras(keras_model)
        y_produced = _evaluate(onnx_model, x)

        self.assertTrue(np.allclose(y_reference, y_produced))

    def _test_one_to_one_operator_core_channels_last(self, keras_model, x):
        '''
        There are two test paths. One is Keras-->CoreML-->ONNX and the other one is Keras-->ONNX.

        Keras-->CoreML-->ONNX:

        Keras computation path:
            [N, C, H, W] ---> numpy transpose ---> [N, H, W, C] ---> keras convolution --->
            [N, H, W, C] ---> numpy transpose ---> [N, C, H, W]

        ONNX computation path:
            [N, C, H, W] ---> ONNX convolution ---> [N, C, H, W]

        The reason for having extra transpose's in the Keras path is that CoreMLTools doesn't not handle channels_last
        flag properly. Precisely, oreMLTools always converts Conv2D under channels_first mode.

        Keras-->ONNX

        Keras computation path:
            [N, C, H, W] ---> numpy transpose ---> [N, H, W, C] ---> keras convolution --->
            [N, H, W, C]

        ONNX computation path:
            [N, C, H, W] ---> numpy transpose ---> [N, H, W, C] ---> ONNX convolution ---> [N, H, W, C]

        '''
        # Verify Keras-to-CoreML-to-ONNX path
        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model_p1 = onnxmltools.convert_coreml(coreml_model)
        onnx_model_p2 = onnxmltools.convert_keras(keras_model)

        if isinstance(x, list):
            x_t = [np.transpose(_, [0, 2, 3, 1]) for _ in x]
        else:
            x_t = np.transpose(x, [0, 2, 3, 1])
        y_reference = np.transpose(keras_model.predict(x_t), [0, 3, 1, 2])
        y_produced = _evaluate(onnx_model_p1, x)

        self.assertTrue(np.allclose(y_reference, y_produced))

        # Verify Keras-to-ONNX path
        y_reference = np.transpose(y_reference, [0, 2, 3, 1])
        y_produced = _evaluate(onnx_model_p2, x_t)

        self.assertTrue(np.allclose(y_reference, y_produced, atol=1e-6))

    def test_dense(self):
        N, C, D = 2, 3, 2
        x = _create_tensor(N, C)

        input = Input(shape=(C,))
        result = Dense(D)(input)
        keras_model = Model(inputs=input, outputs=result)
        keras_model.compile(optimizer='adagrad', loss='mse')

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        y_reference = keras_model.predict(x)
        y_produced = _evaluate(onnx_model, x).reshape(N, D)

        self.assertTrue(np.allclose(y_reference, y_produced))

    def test_dense_with_dropout(self):
        N, C, D = 2, 3, 2
        x = _create_tensor(N, C)

        input = Input(shape=(C,))
        hidden = Dense(D, activation='relu')(input)
        result = Dropout(0.2)(hidden)

        keras_model = Model(inputs=input, outputs=result)
        keras_model.compile(optimizer='sgd', loss='mse')

        self._test_one_to_one_operator_core_keras(keras_model, x)

    def test_conv_4d(self):
        N, C, H, W = 1, 2, 4, 3
        x = _create_tensor(N, C, H, W)

        input = Input(shape=(H, W, C))
        result = Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                        data_format='channels_last')(input)
        model = Model(inputs=input, outputs=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x)

    def test_pooling_4d(self):
        layers_to_be_tested = [MaxPooling2D, AveragePooling2D]
        N, C, H, W = 1, 2, 4, 3
        x = _create_tensor(N, C, H, W)
        for layer in layers_to_be_tested:
            input = Input(shape=(H, W, C))
            result = layer(2, data_format='channels_last')(input)
            model = Model(inputs=input, outputs=result)
            model.compile(optimizer='adagrad', loss='mse')

            self._test_one_to_one_operator_core_channels_last(model, x)

    @unittest.skip('Skip because CNTK is not able to evaluate this model')
    def test_convolution_transpose_2d(self):
        N, C, H, W = 2, 2, 1, 1
        x = _create_tensor(N, C, H, W)

        input = Input(shape=(H, W, C))
        result = Conv2DTranspose(2, (2, 1), data_format='channels_last')(input)
        model = Model(inputs=input, outputs=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x)

    def test_merge_2d(self):
        # Skip Concatenate for now because  CoreML Concatenate needs 4-D input
        layers_to_be_tested = [Add, Maximum, Multiply, Average, Dot]
        N, C = 2, 3
        x1 = _create_tensor(N, C)
        x2 = _create_tensor(N, C)
        for layer in layers_to_be_tested:
            input1 = Input(shape=(C,))
            input2 = Input(shape=(C,))
            if layer == Dot:
                result = layer(axes=-1)([input1, input2])
            else:
                result = layer()([input1, input2])
            model = Model(inputs=[input1, input2], outputs=result)
            model.compile(optimizer='adagrad', loss='mse')
            self._test_one_to_one_operator_core(model, [x1, x2])

    def test_merge_4d(self):
        layers_to_be_tested = [Add, Maximum, Multiply, Average, Concatenate]
        N, C, H, W = 2, 2, 1, 3
        x1 = _create_tensor(N, C, H, W)
        x2 = _create_tensor(N, C, H, W)
        for layer in layers_to_be_tested:
            input1 = Input(shape=(H, W, C))
            input2 = Input(shape=(H, W, C))
            output = layer()([input1, input2])
            model = Model(inputs=[input1, input2], outputs=output)
            model.compile(optimizer='adagrad', loss='mse')
            self._test_one_to_one_operator_core_channels_last(model, [x1, x2])

    def test_activation_2d(self):
        activation_to_be_tested = ['tanh', 'relu', 'sigmoid', 'softsign', 'elu', 'softplus', LeakyReLU]
        N, C = 2, 3
        x = _create_tensor(N, C)

        for activation in activation_to_be_tested:
            input = Input(shape=(C,))
            if isinstance(activation, str):
                result = Activation(activation)(input)
            else:
                result = activation()(input)
            model = Model(inputs=input, outputs=result)
            model.compile(optimizer='adagrad', loss='mse')

            self._test_one_to_one_operator_core(model, x)

    def test_activation_4d(self):
        activation_to_be_tested = ['tanh', 'relu', 'sigmoid', 'softsign', 'elu', 'softplus', LeakyReLU]

        N, C, H, W = 2, 3, 4, 5
        x = _create_tensor(N, C, H, W)

        for activation in activation_to_be_tested:
            input = Input(shape=(H, W, C))
            if isinstance(activation, str):
                result = Activation(activation)(input)
            else:
                result = activation()(input)
            model = Model(inputs=input, outputs=result)
            model.compile(optimizer='adagrad', loss='mse')

            self._test_one_to_one_operator_core_channels_last(model, x)

    def test_embedding(self):
        # This test is active only for Caffe2
        if _find_backend() == 'cntk':
            return 0
        low, high = 0, 3
        x = np.random.randint(low=low, high=high, size=2, dtype='int64')
        model = Sequential()
        model.add(Embedding(high - low + 1, 2, input_length=1))
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core(model, x)

    def test_batch_normalization(self):
        # This test is active only for CNTK
        if _find_backend() == 'caffe2':
            return 0
        N, C, H, W = 2, 2, 3, 4
        x = _create_tensor(N, C, H, W)
        model = Sequential()
        input = Input(shape=(H, W, C))
        result = BatchNormalization(beta_initializer='random_uniform', gamma_initializer='random_uniform',
                                    moving_mean_initializer='random_uniform',
                                    moving_variance_initializer=RandomUniform(minval=0.1, maxval=0.5),
                                    )(input)
        model = Model(inputs=input, outputs=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x)

    @unittest.skip('This is not supported by either CNTK nor Caffe2')
    def test_gru(self):
        N, T, C, D = 1, 2, 3, 2
        input = Input(shape=(T, C))
        rnn = GRU(D, recurrent_activation='sigmoid')
        result = rnn(input)
        model = Model([input], [result])
        model.compile(optimizer='adagrad', loss='mse')

        x = np.random.rand(N, T, C)
        self._test_one_to_one_operator_core(model, x)

    def test_upsample(self):
        # This test is only active for Caffe2
        if _find_backend() == 'cntk':
            return

        N, C, H, W = 2, 3, 1, 2
        x = _create_tensor(N, C, H, W)

        input = Input(shape=(H, W, C))
        result = UpSampling2D(input)
        model = Model(inputs=input, outputs=result)
        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x)

    def test_flatten(self):
        if StrictVersion(onnx.__version__) >= StrictVersion('1.2'):
            # The latest CNTK release does not support the new ONNX Reshape
            return 0
        N, C, H, W, D = 2, 3, 1, 2, 2
        x = _create_tensor(N, C, H, W)

        keras_model = Sequential()
        keras_model.add(Flatten(input_shape=(H, W, C)))
        keras_model.add(Dense(D))
        keras_model.compile(optimizer='adagrad', loss='mse')

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)

        y_reference = keras_model.predict(np.transpose(x, [0, 2, 3, 1]))

        y_produced = _evaluate(onnx_model, x).reshape(N, D)

        self.assertTrue(np.allclose(y_reference, y_produced))

    def test_reshape(self):
        if _find_backend() == 'cntk':
            return 0
        N, C, H, W = 2, 3, 1, 2
        x = _create_tensor(N, C, H, W)

        keras_model = Sequential()
        keras_model.add(Reshape((1, C * H * W, 1), input_shape=(H, W, C)))
        keras_model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(keras_model, x)

    def test_sequential_model_with_multiple_operators(self):
        N, C, H, W = 2, 3, 5, 5
        x = _create_tensor(N, C, H, W)

        model = Sequential()
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

        model.compile(optimizer='adagrad', loss='mse')

        self._test_one_to_one_operator_core_channels_last(model, x)

    def test_recursive_model(self):
        N, C, D = 2, 3, 3
        x = _create_tensor(N, C)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_model1 = Model(inputs=sub_input1, outputs=sub_mapped1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = Dense(D)(sub_input2)
        sub_model2 = Model(inputs=sub_input2, outputs=sub_mapped2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_2 = sub_model1(input1)
        mapped2_2 = sub_model2(input2)
        sub_sum = Add()([mapped1_2, mapped2_2])
        keras_model = Model(inputs=[input1, input2], output=sub_sum)

        coreml_model = coremltools.converters.keras.convert(keras_model)
        onnx_model = onnxmltools.convert_coreml(coreml_model)
        x = [x, 2*x]
        y_reference = keras_model.predict(x)
        y_produced = _evaluate(onnx_model, x).reshape(N, D)

    def test_recursive_and_shared_model(self):
        N, C, D = 2, 3, 3
        x = _create_tensor(N, C)

        sub_input1 = Input(shape=(C,))
        sub_mapped1 = Dense(D)(sub_input1)
        sub_output1 = Activation('sigmoid')(sub_mapped1)
        sub_model1 = Model(inputs=sub_input1, outputs=sub_output1)

        sub_input2 = Input(shape=(C,))
        sub_mapped2 = sub_model1(sub_input2)
        sub_output2 = Activation('tanh')(sub_mapped2)
        sub_model2 = Model(inputs=sub_input2, outputs=sub_output2)

        input1 = Input(shape=(D,))
        input2 = Input(shape=(D,))
        mapped1_1 = Activation('tanh')(input1)
        mapped2_1 = Activation('sigmoid')(input2)
        mapped1_2 = sub_model1(mapped1_1)
        mapped1_3 = sub_model1(mapped1_2)
        mapped2_2 = sub_model2(mapped2_1)
        sub_sum = Add()([mapped1_3, mapped2_2])
        model = Model(inputs=[input1, input2], outputs=sub_sum)

        self._test_one_to_one_operator_core_keras(model, [x, 2 * x])
