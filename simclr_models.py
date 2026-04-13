import tensorflow as tf
import numpy as np

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def create_base_model(input_shape, model_name="base_model"):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.

    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    return tf.keras.Model(inputs, x, name=model_name)

def create_sincnet_base_model(
    input_shape,
    model_name: str = "sincnet_base_model",
    num_sinc_filters: int = 32,
    sinc_kernel_size: int = 25,
    sample_rate: float = 50.0,
    depthwise: bool = True,
):
    """
    Base encoder for SimCLR on MotionSense, with a SincNet frontend.

    Architecture
    ------------
    Input  (window_size, num_channels)
        -> SincConv1D "sincconv"
           depthwise=True:  (window_size, num_channels * num_sinc_filters)
           depthwise=False: (window_size, num_sinc_filters)
        -> BatchNormalization
        -> LeakyReLU
        -> Conv1D: 64 filters, kernel 16, ReLU, L2
        -> Dropout 10%
        -> Conv1D: 96 filters, kernel  8, ReLU, L2
        -> Dropout 10%
        -> GlobalMaxPool1D

    Parameters
    ----------
    input_shape : tuple
        (window_size, num_channels), e.g. (400, 3) for MotionSense acc-only.
    model_name : str
    num_sinc_filters : int
        Filters per channel (depthwise=True) or total filters (depthwise=False).
    sinc_kernel_size : int
        FIR filter length.  25 samples at 50 Hz ~ 0.5 s.
    sample_rate : float
        Acquisition frequency in Hz.
    depthwise : bool
        Passed through to SincConv1D.

    Returns
    -------
    model : tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    x = SincConv1D(
        num_filters=num_sinc_filters,
        kernel_size=sinc_kernel_size,
        sample_rate=sample_rate,
        depthwise=depthwise,
        name="sincconv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(
        64, 16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
        96, 8,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.GlobalMaxPool1D(
        data_format="channels_last", name="global_max_pooling1d"
    )(x)

    return tf.keras.Model(inputs, x, name=model_name)


def attach_simclr_head(base_model, hidden_1=256, hidden_2=128, hidden_3=50):
    """
    Attach a 3-layer fully-connected encoding head

    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """
    
    input = base_model.input
    x = base_model.output

    projection_1 = tf.keras.layers.Dense(hidden_1)(x)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    simclr_model = tf.keras.Model(input, projection_3, name= base_model.name + "_simclr")

    return simclr_model


def create_linear_model_from_base_model(base_model, output_shape, intermediate_layer=7):

    """
    Create a linear classification model from the base mode, using activitations from an intermediate layer

    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = base_model.inputs
    x = base_model.layers[intermediate_layer].output
    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model.name + "linear")

    for layer in model.layers[:intermediate_layer+1]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def create_full_classification_model_from_base_model(base_model, output_shape, model_name="TPN", intermediate_layer=7, last_freeze_layer=4):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing

    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        model_name
            name of the output model

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    # inputs = base_model.inputs
    intermediate_x = base_model.layers[intermediate_layer].output

    x = tf.keras.layers.Dense(1024, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[last_freeze_layer+1:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    """
    Create an intermediate model from base mode, which outputs embeddings of the intermediate layer

    Parameters:
        base_model
            the base model from which the intermediate model is built
        
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model






class SincConv1D(tf.keras.layers.Layer):
    """
    Learnable sinc-based bandpass filter bank for multi-channel time series.

    Each filter is defined by two learnable parameters (lower cutoff f1 and
    bandwidth b), enforcing a bandpass shape:
        h[n] = 2*f2*sinc(2*f2*n) - 2*f1*sinc(2*f1*n),  f2 = f1 + |b|

    Two modes are supported via the `depthwise` argument:

    depthwise=True  (default)
        One independent filter bank per input channel.  Each channel is
        filtered separately and the outputs are concatenated along the
        feature axis.
        Input  (batch, time, C)  ->  output  (batch, time, C * num_filters)

    depthwise=False
        A single shared filter bank applied across all channels jointly.
        Input  (batch, time, C)  ->  output  (batch, time, num_filters)

    In both modes the layer is named "sincconv" in the model graph, so
    plot_sincnet_filter_response always uses sincconv_layer_names=["sincconv"].
    Switching between modes is a single argument change; no other code changes.

    Reference
    ---------
    Ravanelli, M., & Bengio, Y. (2018).
        Speaker Recognition from raw waveform with SincNet.
        https://arxiv.org/abs/1808.00158

    Parameters
    ----------
    num_filters : int
        Filters per channel (depthwise=True) or total filters (depthwise=False).
    kernel_size : int
        FIR filter length (forced odd for linear-phase symmetry).
    sample_rate : float
        Sampling frequency in Hz (50.0 for MotionSense).
    min_low_hz : float
        Hard lower bound on f1 (Hz).
    min_band_hz : float
        Hard lower bound on bandwidth (Hz), ensuring f2 > f1.
    depthwise : bool
        If True (default), one filter bank per input channel.
        If False, one shared filter bank across all channels.
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        sample_rate: float = 50.0,
        min_low_hz: float = 0.1,
        min_band_hz: float = 0.5,
        depthwise: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.min_low_hz  = min_low_hz
        self.min_band_hz = min_band_hz
        self.depthwise   = depthwise

    def build(self, input_shape):
        num_channels = int(input_shape[-1])
        self._num_channels = num_channels

        nyquist   = self.sample_rate / 2.0
        hz_points = np.linspace(
            self.min_low_hz,
            nyquist - (self.min_low_hz + self.min_band_hz),
            self.num_filters + 1,
        )
        f1_init   = (hz_points[:-1] / nyquist).astype(np.float32)
        band_init = ((hz_points[1:] - hz_points[:-1]) / nyquist).astype(np.float32)

        if self.depthwise:
            # Shape (num_channels, num_filters): independent bank per channel
            self.f1_   = self.add_weight(
                name="f1",
                shape=(num_channels, self.num_filters),
                initializer=tf.constant_initializer(
                    np.tile(f1_init, (num_channels, 1))
                ),
                trainable=True,
            )
            self.band_ = self.add_weight(
                name="band",
                shape=(num_channels, self.num_filters),
                initializer=tf.constant_initializer(
                    np.tile(band_init, (num_channels, 1))
                ),
                trainable=True,
            )
        else:
            # Shape (num_filters,): single shared bank
            self.f1_   = self.add_weight(
                name="f1",
                shape=(self.num_filters,),
                initializer=tf.constant_initializer(f1_init),
                trainable=True,
            )
            self.band_ = self.add_weight(
                name="band",
                shape=(self.num_filters,),
                initializer=tf.constant_initializer(band_init),
                trainable=True,
            )

        half = (self.kernel_size - 1) // 2
        self.n_      = tf.cast(tf.range(-half, half + 1), tf.float32)
        self.window_ = tf.signal.hamming_window(self.kernel_size, periodic=False)

        super().build(input_shape)

    def _make_filters(self, f1_norm, band_norm):
        """
        Build normalised windowed-sinc filters.

        Parameters
        ----------
        f1_norm, band_norm : tf.Tensor  shape (num_filters,)
            Normalised (fraction of Nyquist) cutoff parameters.

        Returns
        -------
        tf.Tensor  shape (kernel_size, 1, num_filters)
        """
        nyquist  = self.sample_rate / 2.0
        min_low  = self.min_low_hz  / nyquist
        min_band = self.min_band_hz / nyquist

        f1 = tf.abs(f1_norm)   + min_low
        f2 = tf.clip_by_value(f1 + tf.abs(band_norm) + min_band, 0.0, 1.0)

        n  = tf.reshape(self.n_, (-1, 1))   # (kernel_size, 1)
        f1 = tf.reshape(f1,      ( 1, -1))  # (1, num_filters)
        f2 = tf.reshape(f2,      ( 1, -1))

        def sinc(x):
            px = np.pi * x
            return tf.where(tf.equal(x, 0.0), tf.ones_like(x), tf.sin(px) / px)

        bp  = 2.0*f2*sinc(2.0*f2*n) - 2.0*f1*sinc(2.0*f1*n)
        bp  = bp * tf.reshape(self.window_, (-1, 1))
        bp  = bp / (tf.reduce_sum(tf.abs(bp), axis=0, keepdims=True) + 1e-8)

        return tf.reshape(bp, (self.kernel_size, 1, self.num_filters))

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.Tensor  shape (batch, time, channels)

        Returns
        -------
        depthwise=True  : tf.Tensor  (batch, time, channels * num_filters)
        depthwise=False : tf.Tensor  (batch, time, num_filters)
        """
        if self.depthwise:
            # self.f1_ / self.band_ : (num_channels, num_filters)
            # Apply per-channel filter bank inside call() — valid on both
            # eager tensors and Keras symbolic tensors (KerasTensor).
            channel_outputs = []
            for ch in range(self._num_channels):
                filters  = self._make_filters(self.f1_[ch], self.band_[ch])
                ch_input = tf.expand_dims(inputs[..., ch], axis=-1)
                channel_outputs.append(
                    tf.nn.conv1d(ch_input, filters, stride=1, padding="SAME")
                )
            return tf.concat(channel_outputs, axis=-1)
        else:
            # self.f1_ / self.band_ : (num_filters,)
            # Tile filters across channels for a standard grouped convolution.
            filters = self._make_filters(self.f1_, self.band_)
            filters_tiled = tf.tile(filters, [1, self._num_channels, 1])
            return tf.nn.conv1d(inputs, filters_tiled, stride=1, padding="SAME")

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            sample_rate=self.sample_rate,
            min_low_hz=self.min_low_hz,
            min_band_hz=self.min_band_hz,
            depthwise=self.depthwise,
        )
        return cfg