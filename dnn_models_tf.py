import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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

    Parameters
    ----------
    out_channels : int
        Number of filters.
    kernel_size : int
        Filter length (forced odd internally).
    sample_rate : int
        Signal sample rate in Hz.
    stride : int
        Convolution stride.
    padding : str
        'valid' or 'same'.
    min_low_hz : float
        Hard floor on each filter's lower cutoff frequency (Hz).
        Prevents filters from drifting toward DC / sensor bias.
    min_band_hz : float
        Hard floor on each filter's bandwidth (Hz).
        Prevents filters collapsing to zero-bandwidth spikes.
    max_high_hz : float or None
        Hard ceiling on each filter's upper cutoff frequency (Hz).
        Defaults to Nyquist (sample_rate / 2). Set to a value below
        Nyquist to restrict the filter bank to a frequency sub-range,
        e.g. max_high_hz=10 for IMU signals where activity information
        lives mostly below 10 Hz.

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 stride=1, padding="valid",
                 min_low_hz=0.5, min_band_hz=1.0, max_high_hz=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.out_channels = out_channels
        self.kernel_size  = kernel_size + (1 - kernel_size % 2)  # force odd
        self.sample_rate  = sample_rate
        self.stride       = stride
        self.padding      = padding.upper()
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz
        self.max_high_hz  = max_high_hz if max_high_hz is not None else sample_rate / 2.0

        # Linear initialisation — uniform spacing from min_low_hz to max_high_hz.
        # Linear is preferred over the original Mel spacing for IMU signals because
        # all relevant movement frequencies sit in a narrow low-frequency range
        # where Mel compression provides no benefit.
        low_hz  = min_low_hz
        high_hz = self.max_high_hz - (min_low_hz + min_band_hz)
        if high_hz <= low_hz:
            raise ValueError(
                f"Frequency range is too narrow: max_high_hz={self.max_high_hz} leaves "
                f"no room for {out_channels} filters with min_low_hz={min_low_hz} and "
                f"min_band_hz={min_band_hz}. Increase max_high_hz or reduce the floors."
            )
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
        self.window_ = tf.constant(self._window_np, dtype=tf.float32)
        self.n_      = tf.constant(self._n_np,      dtype=tf.float32)
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
        low  = self.min_low_hz + tf.abs(self.low_hz_)                   # (C, 1)
        high = tf.clip_by_value(
            low + self.min_band_hz + tf.abs(self.band_hz_),
            self.min_low_hz, self.max_high_hz)                           # (C, 1)
        band = tf.squeeze(high - low, axis=1)                            # (C,)

        f_times_t_low  = tf.matmul(low,  self.n_)                       # (C, kernel//2)
        f_times_t_high = tf.matmul(high, self.n_)                       # (C, kernel//2)

        band_pass_left = (
            (tf.sin(f_times_t_high) - tf.sin(f_times_t_low)) /
            (self.n_ / 2.0)
        ) * self.window_                                                  # (C, kernel//2)

        band_pass_center = 2.0 * tf.reshape(band, (-1, 1))               # (C, 1)
        band_pass_right  = tf.reverse(band_pass_left, axis=[1])          # (C, kernel//2)

        band_pass = tf.concat(
            [band_pass_left, band_pass_center, band_pass_right], axis=1) # (C, kernel)
        band_pass = band_pass / (2.0 * band[:, None])

        # (kernel_size, 1, out_channels) for tf.nn.conv1d
        filters = tf.transpose(
            tf.reshape(band_pass, (self.out_channels, self.kernel_size, 1)),
            perm=[1, 2, 0])

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
            "max_high_hz":  self.max_high_hz,
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

        self.input_dim            = int(options["input_dim"])
        self.fc_lay               = options["fc_lay"]
        self.fc_drop              = options["fc_drop"]
        self.fc_use_batchnorm     = options["fc_use_batchnorm"]
        self.fc_use_laynorm       = options["fc_use_laynorm"]
        self.fc_use_laynorm_inp   = options["fc_use_laynorm_inp"]
        self.fc_use_batchnorm_inp = options["fc_use_batchnorm_inp"]
        self.fc_act               = options["fc_act"]

        self.N_fc_lay = len(self.fc_lay)

        self.ln0 = LayerNorm(self.input_dim) if self.fc_use_laynorm_inp  else None
        self.bn0 = layers.BatchNormalization(momentum=0.95) \
                   if self.fc_use_batchnorm_inp else None

        self.wx_layers   = []
        self.bn_layers   = []
        self.ln_layers   = []
        self.act_layers  = []
        self.drop_layers = []

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
# SincNet (subclassed model — kept for backward compatibility)
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
            self.ln_layers.append(LayerNorm(N_filt * out_len))
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
        if self.ln0 is not None:
            x = self.ln0(x)
        if self.bn0 is not None:
            x = self.bn0(x, training=training)

        x = tf.expand_dims(x, axis=-1)

        for i in range(self.N_cnn_lay):
            x = self.conv_layers[i](x)

            if i == 0:
                x = tf.abs(x)

            pool = self.cnn_max_pool_len[i]
            x = tf.nn.max_pool1d(x, ksize=pool, strides=pool, padding="VALID")

            if self.cnn_use_laynorm[i]:
                batch = tf.shape(x)[0]
                x_flat = tf.reshape(x, (batch, -1))
                x_flat = self.ln_layers[i](x_flat)
                x = tf.reshape(x_flat, tf.shape(x))
                x = self.drop_layers[i](self.act_layers[i](x), training=training)
            elif self.cnn_use_batchnorm[i]:
                x = self.bn_layers[i](x, training=training)
                x = self.drop_layers[i](self.act_layers[i](x), training=training)
            else:
                x = self.drop_layers[i](self.act_layers[i](x), training=training)

        batch = tf.shape(x)[0]
        return tf.reshape(x, (batch, -1))


# ---------------------------------------------------------------------------
# Filter response visualisation
# ---------------------------------------------------------------------------

def plot_sincnet_filter_response(model, fs, sincconv_layer_names, n_freqs=1000,
                                 smooth_sigma=10):
    """
    Plot the cumulative frequency response of learned SincNet filters,
    reproducing Fig. 3 from Ravanelli & Bengio (2018).

    Parameters
    ----------
    model                : tf.keras.Model
    fs                   : int    — sample rate in Hz
    sincconv_layer_names : list[str] — layer names to plot,
                           e.g. ["sincconv"] or ["sincconv_ch0", ..., "sincconv_ch2"]
    n_freqs              : int    — frequency axis resolution
    smooth_sigma         : float  — Gaussian smoothing sigma (set to 0 to disable)
    """
    freqs  = np.linspace(0, fs / 2, n_freqs)
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    plt.figure(figsize=(10, 5))

    for idx, layer_name in enumerate(sincconv_layer_names):
        layer = model.get_layer(layer_name)

        low_hz  = layer.min_low_hz  + np.abs(layer.low_hz_.numpy())     # (N_filt, 1)
        band_hz = layer.min_band_hz + np.abs(layer.band_hz_.numpy())     # (N_filt, 1)
        high_hz = np.clip(low_hz + band_hz, layer.min_low_hz,
                          layer.max_high_hz)                             # (N_filt, 1)

        freqs_row = freqs.reshape(1, -1)
        in_band   = (freqs_row >= low_hz) & (freqs_row <= high_hz)

        cumulative = in_band.sum(axis=0).astype(float)
        cumulative /= cumulative.max()

        if smooth_sigma > 0:
            cumulative = gaussian_filter1d(cumulative, sigma=smooth_sigma)

        plt.plot(freqs, cumulative,
                 color=colors[idx % len(colors)],
                 linestyle='-' if idx == 0 else '--',
                 linewidth=2,
                 label=layer_name if len(sincconv_layer_names) > 1 else 'SincNet')

    plt.xlabel('Frequency [Hz]', fontsize=13)
    plt.ylabel('Normalized Filter Sum', fontsize=13)
    plt.title('Cumulative frequency response of the SincNet filters', fontsize=14)
    plt.legend(fontsize=11)
    plt.xlim([0, fs / 2])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('sincnet_filter_response.png', dpi=150, bbox_inches='tight')
    plt.show()

def monitor_sincnet_filters(model, fs, layer_names, epoch):
    """Call this periodically during or after training."""
    fig, axes = plt.subplots(len(layer_names), 3, figsize=(15, 4 * len(layer_names)))
    if len(layer_names) == 1:
        axes = [axes]

    for row, layer_name in enumerate(layer_names):
        layer = model.get_layer(layer_name)

        low_hz  = (layer.min_low_hz  + np.abs(layer.low_hz_.numpy())).flatten()
        band_hz = (layer.min_band_hz + np.abs(layer.band_hz_.numpy())).flatten()
        high_hz = np.clip(low_hz + band_hz, layer.min_low_hz,
                          layer.max_high_hz).flatten()
        bandwidth = high_hz - low_hz

        axes[row][0].hist(low_hz,    bins=20, color='blue',  alpha=0.7)
        axes[row][0].set_title(f'{layer_name} — Low Cutoff [Hz]')

        axes[row][1].hist(high_hz,   bins=20, color='red',   alpha=0.7)
        axes[row][1].set_title(f'{layer_name} — High Cutoff [Hz]')

        axes[row][2].hist(bandwidth, bins=20, color='green', alpha=0.7)
        axes[row][2].set_title(f'{layer_name} — Bandwidth [Hz]')

        print(f"[Epoch {epoch}] {layer_name} | "
              f"low: {low_hz.mean():.2f}±{low_hz.std():.2f} Hz | "
              f"high: {high_hz.mean():.2f}±{high_hz.std():.2f} Hz | "
              f"bw: {bandwidth.mean():.2f}±{bandwidth.std():.2f} Hz")

    plt.suptitle(f'SincNet filter distribution — Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'sincnet_filters_epoch{epoch}.png', dpi=150)
    plt.close()