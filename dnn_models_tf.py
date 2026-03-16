import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def act_fun(act_type):
    """Return a Keras activation layer matching the original PyTorch version."""
    mapping = {
        "relu":       layers.ReLU(),
        "tanh":       layers.Activation("tanh"),
        "sigmoid":    layers.Activation("sigmoid"),
        "leaky_relu": layers.LeakyReLU(negative_slope=0.2),
        "elu":        layers.ELU(),
        "softmax":    layers.Activation("log_softmax"),   # mirrors nn.LogSoftmax
        "linear":     layers.LeakyReLU(negative_slope=1.0),  # same quirk as original
    }
    if act_type not in mapping:
        raise ValueError(f"Unknown activation: {act_type}")
    return mapping[act_type]


# ---------------------------------------------------------------------------
# Layer Normalisation (custom – mirrors the original per-sample LayerNorm)
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="dnn_models_tf")
class LayerNorm(layers.Layer):
    """Layer normalisation over the last dimension, with learnable gamma/beta."""

    def __init__(self, features, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=(self.features,),
            initializer="ones", trainable=True)
        self.beta = self.add_weight(
            name="beta", shape=(self.features,),
            initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"features": self.features, "eps": self.eps})
        return cfg


# ---------------------------------------------------------------------------
# SincConv  (fast, vectorised version)
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="dnn_models_tf")
class SincConv_fast(layers.Layer):
    """
    Sinc-based bandpass filter-bank convolution layer.

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 stride=1, padding="valid",
                 min_low_hz=0.1, min_band_hz=5, **kwargs):
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.kernel_size  = kernel_size + (1 - kernel_size % 2)  # force odd
        self.sample_rate  = sample_rate
        self.stride       = stride
        self.padding      = padding.upper()
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Linear initialisation
        low_hz  = 0.1
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        hz = np.linspace(low_hz, high_hz, out_channels + 1)

        self._init_low_hz  = hz[:-1].reshape(-1, 1).astype(np.float32)
        self._init_band_hz = np.diff(hz).reshape(-1, 1).astype(np.float32)

        # Hamming window (half)
        n_lin = np.linspace(0, self.kernel_size / 2 - 1,
                            int(self.kernel_size / 2), dtype=np.float32)
        self._window_np = (0.54 - 0.46 * np.cos(
            2.0 * math.pi * n_lin / self.kernel_size)).astype(np.float32)

        # Time axis (half, excluding centre)
        n = (self.kernel_size - 1) / 2.0
        self._n_np = (2.0 * math.pi *
                      np.arange(-n, 0, dtype=np.float32) /
                      sample_rate).reshape(1, -1)  # (1, kernel_size//2)

    def build(self, input_shape):
        self.low_hz_ = self.add_weight(
            name="low_hz", shape=(self.out_channels, 1),
            initializer=tf.constant_initializer(self._init_low_hz),
            trainable=True)
        self.band_hz_ = self.add_weight(
            name="band_hz", shape=(self.out_channels, 1),
            initializer=tf.constant_initializer(self._init_band_hz),
            trainable=True)
        self.window_ = tf.constant(self._window_np, dtype=tf.float32)  # (kernel//2,)
        self.n_      = tf.constant(self._n_np,      dtype=tf.float32)  # (1, kernel//2)
        super().build(input_shape)

    def call(self, waveforms):
        """
        Parameters
        ----------
        waveforms : Tensor  (batch, n_samples, 1)   — TF conv format

        Returns
        -------
        Tensor  (batch, n_samples_out, out_channels)
        """
        low  = self.min_low_hz  + tf.abs(self.low_hz_)                              # (C,1)
        high = tf.clip_by_value(
            low + self.min_band_hz + tf.abs(self.band_hz_),
            self.min_low_hz, self.sample_rate / 2.0)
        band = tf.squeeze(high - low, axis=1)                                        # (C,)

        f_times_t_low  = tf.matmul(low,  self.n_)   # (C, kernel//2)
        f_times_t_high = tf.matmul(high, self.n_)   # (C, kernel//2)

        # Equation 4 of the paper, expanded
        band_pass_left = (
            (tf.sin(f_times_t_high) - tf.sin(f_times_t_low)) /
            (self.n_ / 2.0)
        ) * self.window_                                                              # (C, kernel//2)

        band_pass_center = 2.0 * tf.reshape(band, (-1, 1))                          # (C, 1)
        band_pass_right  = tf.reverse(band_pass_left, axis=[1])                     # (C, kernel//2)

        band_pass = tf.concat(
            [band_pass_left, band_pass_center, band_pass_right], axis=1)            # (C, kernel)
        band_pass = band_pass / (2.0 * band[:, None])                               # normalise

        # Reshape to (kernel_size, 1, out_channels) for tf.nn.conv1d
        filters = tf.transpose(
            tf.reshape(band_pass, (self.out_channels, self.kernel_size, 1)),
            perm=[1, 2, 0])                                                          # (kernel,1,C)

        return tf.nn.conv1d(waveforms, filters,
                            stride=self.stride, padding=self.padding)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "out_channels": self.out_channels,
            "kernel_size":  self.kernel_size,
            "sample_rate":  self.sample_rate,
            "stride":       self.stride,
            "padding":      self.padding,
            "min_low_hz":   self.min_low_hz,
            "min_band_hz":  self.min_band_hz,
        })
        return cfg


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(keras.Model):
    """
    Multi-layer perceptron matching the original PyTorch MLP.

    options keys
    ------------
    input_dim, fc_lay, fc_drop, fc_use_batchnorm, fc_use_laynorm,
    fc_use_laynorm_inp, fc_use_batchnorm_inp, fc_act
    """

    def __init__(self, options, **kwargs):
        super().__init__(**kwargs)

        self.input_dim           = int(options["input_dim"])
        self.fc_lay              = options["fc_lay"]
        self.fc_drop             = options["fc_drop"]
        self.fc_use_batchnorm    = options["fc_use_batchnorm"]
        self.fc_use_laynorm      = options["fc_use_laynorm"]
        self.fc_use_laynorm_inp  = options["fc_use_laynorm_inp"]
        self.fc_use_batchnorm_inp= options["fc_use_batchnorm_inp"]
        self.fc_act              = options["fc_act"]

        self.N_fc_lay = len(self.fc_lay)

        # Optional input normalisation
        self.ln0 = LayerNorm(self.input_dim) if self.fc_use_laynorm_inp  else None
        self.bn0 = layers.BatchNormalization(momentum=0.95) \
                   if self.fc_use_batchnorm_inp else None

        self.wx_layers  = []
        self.bn_layers  = []
        self.ln_layers  = []
        self.act_layers = []
        self.drop_layers= []

        current_input = self.input_dim
        for i in range(self.N_fc_lay):
            use_bias = not (self.fc_use_laynorm[i] or self.fc_use_batchnorm[i])
            fan_sum  = current_input + self.fc_lay[i]
            limit    = math.sqrt(0.01 / fan_sum)

            fc = layers.Dense(
                self.fc_lay[i], use_bias=use_bias,
                kernel_initializer=keras.initializers.RandomUniform(-limit, limit),
                bias_initializer="zeros")
            self.wx_layers.append(fc)
            self.ln_layers.append(LayerNorm(self.fc_lay[i]))
            self.bn_layers.append(layers.BatchNormalization(momentum=0.95))
            self.act_layers.append(act_fun(self.fc_act[i]))
            self.drop_layers.append(layers.Dropout(rate=self.fc_drop[i]))

            current_input = self.fc_lay[i]

    def call(self, x, training=False):
        if self.ln0 is not None:
            x = self.ln0(x)
        if self.bn0 is not None:
            x = self.bn0(x, training=training)

        for i in range(self.N_fc_lay):
            if self.fc_act[i] != "linear":
                if self.fc_use_laynorm[i]:
                    x = self.drop_layers[i](
                        self.act_layers[i](self.ln_layers[i](self.wx_layers[i](x))),
                        training=training)
                elif self.fc_use_batchnorm[i]:
                    x = self.drop_layers[i](
                        self.act_layers[i](self.bn_layers[i](self.wx_layers[i](x),
                                                             training=training)),
                        training=training)
                else:
                    x = self.drop_layers[i](
                        self.act_layers[i](self.wx_layers[i](x)),
                        training=training)
            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop_layers[i](
                        self.ln_layers[i](self.wx_layers[i](x)),
                        training=training)
                elif self.fc_use_batchnorm[i]:
                    x = self.drop_layers[i](
                        self.bn_layers[i](self.wx_layers[i](x), training=training),
                        training=training)
                else:
                    x = self.drop_layers[i](self.wx_layers[i](x), training=training)
        return x


# ---------------------------------------------------------------------------
# SincNet
# ---------------------------------------------------------------------------

class SincNet(keras.Model):
    """
    SincNet model (CNN front-end + configurable stack).

    options keys
    ------------
    cnn_N_filt, cnn_len_filt, cnn_max_pool_len, cnn_act, cnn_drop,
    cnn_use_laynorm, cnn_use_batchnorm, cnn_use_laynorm_inp,
    cnn_use_batchnorm_inp, input_dim, fs
    """

    def __init__(self, options, **kwargs):
        super().__init__(**kwargs)

        self.cnn_N_filt            = options["cnn_N_filt"]
        self.cnn_len_filt          = options["cnn_len_filt"]
        self.cnn_max_pool_len      = options["cnn_max_pool_len"]
        self.cnn_act               = options["cnn_act"]
        self.cnn_drop              = options["cnn_drop"]
        self.cnn_use_laynorm       = options["cnn_use_laynorm"]
        self.cnn_use_batchnorm     = options["cnn_use_batchnorm"]
        self.cnn_use_laynorm_inp   = options["cnn_use_laynorm_inp"]
        self.cnn_use_batchnorm_inp = options["cnn_use_batchnorm_inp"]
        self.input_dim             = int(options["input_dim"])
        self.fs                    = options["fs"]
        self.N_cnn_lay             = len(self.cnn_N_filt)

        # Optional input normalisation
        self.ln0 = LayerNorm(self.input_dim) if self.cnn_use_laynorm_inp  else None
        self.bn0 = layers.BatchNormalization(momentum=0.95) \
                   if self.cnn_use_batchnorm_inp else None

        self.conv_layers = []
        self.bn_layers   = []
        self.ln_layers   = []
        self.act_layers  = []
        self.drop_layers = []

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):
            N_filt   = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])
            pool_len = int(self.cnn_max_pool_len[i])

            self.drop_layers.append(layers.Dropout(rate=self.cnn_drop[i]))
            self.act_layers.append(act_fun(self.cnn_act[i]))

            out_len = int((current_input - len_filt + 1) / pool_len)
            self.ln_layers.append(LayerNorm(N_filt * out_len))  # flattened dim for LN
            self.bn_layers.append(layers.BatchNormalization(momentum=0.95))

            if i == 0:
                self.conv_layers.append(
                    SincConv_fast(N_filt, len_filt, self.fs))
            else:
                self.conv_layers.append(
                    layers.Conv1D(N_filt, len_filt, padding="valid"))

            current_input = out_len

        self.out_dim = current_input * int(self.cnn_N_filt[-1])

    def call(self, x, training=False):
        """
        Parameters
        ----------
        x : Tensor  (batch, seq_len)

        Returns
        -------
        Tensor  (batch, out_dim)
        """
        if self.ln0 is not None:
            x = self.ln0(x)
        if self.bn0 is not None:
            x = self.bn0(x, training=training)

        # (batch, seq_len) → (batch, seq_len, 1)
        x = tf.expand_dims(x, axis=-1)

        for i in range(self.N_cnn_lay):
            x = self.conv_layers[i](x)

            # Absolute value on first SincConv layer (mirrors torch.abs in original)
            if i == 0:
                x = tf.abs(x)

            # Max-pooling  (TF: input shape (batch, steps, channels))
            pool = self.cnn_max_pool_len[i]
            x = tf.nn.max_pool1d(x, ksize=pool, strides=pool, padding="VALID")

            if self.cnn_use_laynorm[i]:
                batch = tf.shape(x)[0]
                x_flat = tf.reshape(x, (batch, -1))
                x_flat = self.ln_layers[i](x_flat)
                x = tf.reshape(x_flat, tf.shape(x))
                x = self.drop_layers[i](self.act_layers[i](x), training=training)

            elif self.cnn_use_batchnorm[i]:
                # BN over channels axis (last dim in TF)
                x = self.bn_layers[i](x, training=training)
                x = self.drop_layers[i](self.act_layers[i](x), training=training)

            else:
                x = self.drop_layers[i](self.act_layers[i](x), training=training)

        batch = tf.shape(x)[0]
        return tf.reshape(x, (batch, -1))
