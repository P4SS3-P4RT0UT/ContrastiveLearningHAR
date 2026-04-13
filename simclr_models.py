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
    sample_rate: int = 50,
):
    """
    Base encoder for SimCLR on MotionSense, with a SincNet frontend.

    The SincNet layer replaces the first Conv1D layer of the original TPN
    architecture (Tang et al., 2020) with learnable sinc-based bandpass
    filters (Ravanelli & Bengio, 2018).  This gives the model an inductive
    bias towards frequency-selective features while keeping the number of
    trainable parameters comparable.

    Architecture
    ------------
    Input  (window_size, num_channels)
        -> SincConv1D per channel (num_sinc_filters, sinc_kernel_size)
           then Concatenate  ->  (window_size, num_channels * num_sinc_filters)
        -> BatchNormalization + LeakyReLU
        -> Conv1D: 64 filters, 16 kernel, L2 regularizer
        -> Dropout 10%
        -> Conv1D: 96 filters,  8 kernel, L2 regularizer
        -> Dropout 10%
        -> GlobalMaxPool1D

    Parameters
    ----------
    input_shape : tuple
        (window_size, num_channels), e.g. (200, 6) for MotionSense.
    model_name : str
    num_sinc_filters : int
        Bandpass filters per IMU channel.  32 gives 192 feature maps after
        6-channel concatenation, similar in capacity to the original 32-filter
        first Conv1D.
    sinc_kernel_size : int
        FIR filter length.  25 samples at 50 Hz ≈ 0.5 s window, which
        captures most relevant motion frequencies (0.5 – 20 Hz).
    sample_rate : int
        Acquisition frequency in Hz (MotionSense = 50 Hz).

    Returns
    -------
    model : tf.keras.Model
    """
    window_size, num_channels = input_shape

    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ---- SincNet frontend ----
    x = _sincnet_frontend(
        inputs,
        num_channels=num_channels,
        num_sinc_filters=num_sinc_filters,
        sinc_kernel_size=sinc_kernel_size,
        sample_rate=sample_rate,
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # ---- Temporal Conv blocks (mirrors TPN layers 2 & 3) ----
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
    SincNet layer: a bank of parameterized sinc-based bandpass filters.

    Each filter is defined by two learnable parameters (low cutoff frequency
    f1 and bandwidth b), enforcing a bandpass shape:
        h[n] = 2*f2*sinc(2*f2*n) - 2*f1*sinc(2*f1*n)

    where f2 = f1 + |b|, ensuring f2 > f1 > 0.

    Reference:
        Ravanelli, M., & Bengio, Y. (2018).
        Speaker Recognition from raw waveform with SincNet.
        https://arxiv.org/abs/1808.00158

    Parameters
    ----------
    num_filters : int
        Number of bandpass filters (output channels).
    kernel_size : int
        Length of each FIR filter. Should be odd for symmetric filters.
    sample_rate : int
        Sampling rate of the input signal in Hz. Used to initialise
        filter cutoff frequencies on a linear scale and to enforce the
        Nyquist constraint.
    min_low_hz : float
        Minimum allowed value for the lower cutoff frequency (Hz).
    min_band_hz : float
        Minimum allowed bandwidth (Hz), ensuring f2 > f1.
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        sample_rate: int = 50,
        min_low_hz: float = 0.1,
        min_band_hz: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Kernel size must be odd for a symmetric (linear-phase) FIR filter.
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

    def build(self, input_shape):

        low_hz = self.min_low_hz
        high_hz = self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)

        # Evenly spaced breakpoints in Hz, then derive f1 / bandwidth
        hz_points = np.linspace(low_hz, high_hz, self.num_filters + 1)

        # f1: lower cutoff; band: bandwidth (f2 = f1 + band)
        f1_init = hz_points[:-1] / (self.sample_rate / 2.0)  # normalised to [0, 1]
        band_init = (hz_points[1:] - hz_points[:-1]) / (self.sample_rate / 2.0)

        self.f1_ = self.add_weight(
            name="f1",
            shape=(self.num_filters,),
            initializer=tf.constant_initializer(f1_init.astype(np.float32)),
            trainable=True,
        )
        self.band_ = self.add_weight(
            name="band",
            shape=(self.num_filters,),
            initializer=tf.constant_initializer(band_init.astype(np.float32)),
            trainable=True,
        )

        # Half-length of filter; time axis (normalised, shape [half_kernel])
        half = (self.kernel_size - 1) // 2
        n_lin = tf.linspace(-half, half, self.kernel_size)  # integer time steps
        self.n_ = tf.cast(n_lin, tf.float32)                # stored as non-trainable

        # Hamming window to reduce spectral leakage
        self.window_ = tf.signal.hamming_window(self.kernel_size, periodic=False)

        super().build(input_shape)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.Tensor, shape (batch, time, 1)
            Single-channel waveform segment.

        Returns
        -------
        tf.Tensor, shape (batch, time', num_filters)
        """
        # Enforce constraints: f1 > min_low_hz, band > min_band_hz
        min_low = self.min_low_hz / (self.sample_rate / 2.0)
        min_band = self.min_band_hz / (self.sample_rate / 2.0)

        f1 = tf.abs(self.f1_) + min_low
        f2 = tf.clip_by_value(f1 + tf.abs(self.band_) + min_band, 0.0, 1.0)

        # Build sinc filters:  h[n] = 2*f2*sinc(2πf2*n) - 2*f1*sinc(2πf1*n)
        # n_ has shape [kernel_size]; f1/f2 have shape [num_filters]
        # We broadcast to [kernel_size, num_filters]
        n = tf.reshape(self.n_, (-1, 1))             # (kernel_size, 1)
        f1_b = tf.reshape(f1, (1, -1))               # (1, num_filters)
        f2_b = tf.reshape(f2, (1, -1))               # (1, num_filters)

        # sinc(x) = sin(π x) / (π x);  at x=0 sinc(0)=1 by definition
        def sinc(x):
            pi_x = np.pi * x
            return tf.where(
                tf.equal(x, 0.0),
                tf.ones_like(x),
                tf.math.sin(pi_x) / pi_x,
            )

        band_pass = (
            2.0 * f2_b * sinc(2.0 * f2_b * tf.cast(n, tf.float32))
            - 2.0 * f1_b * sinc(2.0 * f1_b * tf.cast(n, tf.float32))
        )  # (kernel_size, num_filters)

        # Apply Hamming window
        window = tf.reshape(self.window_, (-1, 1))   # (kernel_size, 1)
        band_pass = band_pass * window

        # Normalise each filter so its L1 norm = 1
        band_pass = band_pass / (
            tf.reduce_sum(tf.abs(band_pass), axis=0, keepdims=True) + 1e-8
        )

        # Reshape for Conv1D weight format: (kernel_size, in_channels=1, num_filters)
        filters = tf.reshape(band_pass, (self.kernel_size, 1, self.num_filters))

        # Apply filters via a depthwise-style 1-D convolution
        out = tf.nn.conv1d(inputs, filters, stride=1, padding="SAME")
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "sample_rate": self.sample_rate,
                "min_low_hz": self.min_low_hz,
                "min_band_hz": self.min_band_hz,
            }
        )
        return config

def _sincnet_frontend(
    x,
    num_channels: int,
    num_sinc_filters: int,
    sinc_kernel_size: int,
    sample_rate: int,
):
    """
    Apply SincConv1D independently to each IMU channel and concatenate.

    MotionSense has 6 channels (acc_x/y/z, gyro_x/y/z); this function
    is parameterised by num_channels so it also works with the 3-channel
    accelerometer-only variant (num_channels=3).  One SincConv1D filter
    bank is applied independently to each channel, and the outputs are
    concatenated along the feature axis.

    Parameters
    ----------
    x : tf.Tensor, shape (batch, window_size, num_channels)
    num_channels : int
    num_sinc_filters : int
        Number of bandpass filters per channel.
    sinc_kernel_size : int
        FIR filter length (will be made odd internally).
    sample_rate : int
        MotionSense is sampled at 50 Hz.

    Returns
    -------
    tf.Tensor, shape (batch, window_size, num_channels * num_sinc_filters)
    """
    channel_outputs = []
    for ch in range(num_channels):
        # Extract channel: (batch, window_size) -> (batch, window_size, 1)
        ch_slice = tf.expand_dims(x[:, :, ch], axis=-1)
        sinc_out = SincConv1D(
            num_filters=num_sinc_filters,
            kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
            name=f"sincconv_ch{ch}",
        )(ch_slice)  # (batch, window_size, num_sinc_filters)
        channel_outputs.append(sinc_out)

    # Concatenate along the feature axis
    return tf.keras.layers.Concatenate(axis=-1)(channel_outputs)