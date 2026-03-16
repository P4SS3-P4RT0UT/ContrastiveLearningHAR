import tensorflow as tf
from dnn_models_tf import SincConv_fast, LayerNorm


# ---------------------------------------------------------------------------
# Serialisable replacements for Lambda layers
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="sincnet_model")
class AbsoluteValue(tf.keras.layers.Layer):
    """
    Element-wise absolute value. Replaces Lambda(tf.abs) to allow safe
    serialisation. Used after SincConv to obtain energy envelopes.
    """

    def call(self, x):
        return tf.abs(x)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_sincnet_model(input_shape, sincnet_options, model_name="sincnet_model"):
    """
    Create a SincNet-based feature extraction model using the Keras functional API.

    A learned 1x1 Conv (channel_mix) projects the N IMU axes into a single
    mixed channel before the SincConv filter bank, allowing cross-axis features
    to be learned from the very first layer. The subsequent Conv1D layers then
    refine these cross-axis frequency representations.

    The model returns a flat feature vector, making it a drop-in replacement
    for other base models such as the TPN Conv1D architecture.

    Reference:
        Ravanelli, M., & Bengio, Y. (2018). Speaker Recognition from raw waveform
        with SincNet. https://arxiv.org/abs/1808.00158

    Architecture:
        Input  (window_size, num_channels)
        -> [Optional] LayerNorm / BatchNorm on input
        -> Conv1D(1, kernel=1)          # learned cross-axis channel mix
        -> SincConv_fast + AbsoluteValue
        -> MaxPool1D + [Norm] + Activation + Dropout
        -> Conv1D  (repeated for remaining CNN layers)
           -> MaxPool1D + [Norm] + Activation + Dropout
        -> Flatten  ->  1-D feature vector

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample: (window_size, num_channels).

    sincnet_options : dict
        Configuration dictionary. All parameters can be set here — nothing
        is hardcoded inside the function.

        CNN architecture:
            cnn_N_filt        : list[int]   - filters per CNN layer
            cnn_len_filt      : list[int]   - kernel size per CNN layer
            cnn_max_pool_len  : list[int]   - max-pool size per CNN layer
            cnn_act           : list[str]   - activation per CNN layer
                                              ('relu', 'tanh', 'leaky_relu', ...)
            cnn_drop          : list[float] - dropout rate per CNN layer
            cnn_use_laynorm   : list[bool]  - use LayerNorm after each CNN layer
            cnn_use_batchnorm : list[bool]  - use BatchNorm after each CNN layer
            cnn_use_laynorm_inp   : bool    - apply LayerNorm to the raw input
            cnn_use_batchnorm_inp : bool    - apply BatchNorm to the raw input

        SincConv filter bank (layer 0 only):
            fs                : int         - signal sample rate (Hz)
            sinc_min_low_hz   : float       - floor on filter lower cutoff (Hz)
                                              default: 0.5
            sinc_min_band_hz  : float       - floor on filter bandwidth (Hz)
                                              default: 1.0
            sinc_max_high_hz  : float|None  - ceiling on filter upper cutoff (Hz)
                                              default: None (= fs/2, i.e. Nyquist)
                                              set e.g. to 10.0 to restrict the
                                              filter bank to 0-10 Hz for IMU

    model_name : str, optional
        Name assigned to the returned tf.keras.Model.

    Returns
    -------
    model : tf.keras.Model
        Functional Keras model.
        Input  shape : (batch, window_size, num_channels)
        Output shape : (batch, out_dim)

    Examples
    --------
    >>> options = {
    ...     "cnn_N_filt":            [80, 60, 60],
    ...     "cnn_len_filt":          [51, 5, 5],
    ...     "cnn_max_pool_len":      [3, 3, 3],
    ...     "cnn_act":               ["leaky_relu", "leaky_relu", "leaky_relu"],
    ...     "cnn_drop":              [0.0, 0.1, 0.1],
    ...     "cnn_use_laynorm":       [True, True, True],
    ...     "cnn_use_batchnorm":     [False, False, False],
    ...     "cnn_use_laynorm_inp":   True,
    ...     "cnn_use_batchnorm_inp": False,
    ...     "fs":                    50,
    ...     "sinc_min_low_hz":       0.5,
    ...     "sinc_min_band_hz":      1.0,
    ...     "sinc_max_high_hz":      10.0,
    ... }
    >>> model = create_sincnet_model(input_shape=(400, 3), sincnet_options=options)
    >>> model.summary()
    """

    window_size, num_channels = input_shape

    # CNN options
    cnn_N_filt            = sincnet_options["cnn_N_filt"]
    cnn_len_filt          = sincnet_options["cnn_len_filt"]
    cnn_max_pool_len      = sincnet_options["cnn_max_pool_len"]
    cnn_act               = sincnet_options["cnn_act"]
    cnn_drop              = sincnet_options["cnn_drop"]
    cnn_use_laynorm       = sincnet_options["cnn_use_laynorm"]
    cnn_use_batchnorm     = sincnet_options["cnn_use_batchnorm"]
    cnn_use_laynorm_inp   = sincnet_options["cnn_use_laynorm_inp"]
    cnn_use_batchnorm_inp = sincnet_options["cnn_use_batchnorm_inp"]

    # SincConv frequency options — .get() provides defaults so existing
    # configs that omit these keys continue to work unchanged
    fs               = sincnet_options["fs"]
    sinc_min_low_hz  = sincnet_options.get("sinc_min_low_hz",  0.5)
    sinc_min_band_hz = sincnet_options.get("sinc_min_band_hz", 1.0)
    sinc_max_high_hz = sincnet_options.get("sinc_max_high_hz", None)  # None -> Nyquist

    N_cnn_lay = len(cnn_N_filt)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs

    # ------------------------------------------------------------------
    # Optional input normalisation
    # LayerNorm flattens the full (window_size, num_channels) feature
    # vector per sample, matching the original's per-utterance normalisation.
    # ------------------------------------------------------------------
    if cnn_use_laynorm_inp:
        x = tf.keras.layers.Reshape((window_size * num_channels,),
                                    name="ln_inp_flatten")(x)
        x = LayerNorm(window_size * num_channels, name="ln_inp")(x)
        x = tf.keras.layers.Reshape(input_shape, name="ln_inp_restore")(x)

    elif cnn_use_batchnorm_inp:
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name="bn_inp")(x)

    # ------------------------------------------------------------------
    # Layer 0 — cross-axis channel mix + SincConv
    #
    # A bias-free 1x1 Conv learns a weighted sum of the IMU axes:
    #   output = w_x*X + w_y*Y + w_z*Z  -> (batch, window_size, 1)
    # This lets the network decide how to combine axes before frequency
    # analysis, enabling cross-axis features from the very first layer.
    #
    # use_bias=False prevents the mixing layer from shifting the signal
    # baseline, which would interfere with sinc filter interpretation.
    # ------------------------------------------------------------------
    N_filt_0   = int(cnn_N_filt[0])
    len_filt_0 = int(cnn_len_filt[0])
    pool_len_0 = int(cnn_max_pool_len[0])

    x = tf.keras.layers.Conv1D(
        1, kernel_size=1,
        use_bias=False,
        name="channel_mix"
    )(x)   # (batch, window_size, num_channels) -> (batch, window_size, 1)

    x = SincConv_fast(
        N_filt_0, len_filt_0, fs,
        padding="valid",
        min_low_hz=sinc_min_low_hz,
        min_band_hz=sinc_min_band_hz,
        max_high_hz=sinc_max_high_hz,
        name="sincconv"
    )(x)

    # |·| rectification — converts signed filter responses to energy envelopes,
    # matching torch.abs() in the original SincNet forward pass
    x = AbsoluteValue(name="abs_sinc")(x)

    x = tf.keras.layers.MaxPool1D(
        pool_size=pool_len_0, strides=pool_len_0,
        padding="valid", name="maxpool_0"
    )(x)
    x = _apply_norm_act_drop(
        x, layer_idx=0,
        use_laynorm=cnn_use_laynorm[0],
        use_batchnorm=cnn_use_batchnorm[0],
        act_type=cnn_act[0],
        drop_rate=cnn_drop[0],
    )

    # ------------------------------------------------------------------
    # Layers 1+ — standard shared Conv1D
    # ------------------------------------------------------------------
    for i in range(1, N_cnn_lay):
        N_filt   = int(cnn_N_filt[i])
        len_filt = int(cnn_len_filt[i])
        pool_len = int(cnn_max_pool_len[i])

        x = tf.keras.layers.Conv1D(
            N_filt, len_filt,
            padding="valid",
            use_bias=not (cnn_use_laynorm[i] or cnn_use_batchnorm[i]),
            name=f"conv1d_{i}"
        )(x)

        x = tf.keras.layers.MaxPool1D(
            pool_size=pool_len, strides=pool_len,
            padding="valid", name=f"maxpool_{i}"
        )(x)

        x = _apply_norm_act_drop(
            x, layer_idx=i,
            use_laynorm=cnn_use_laynorm[i],
            use_batchnorm=cnn_use_batchnorm[i],
            act_type=cnn_act[i],
            drop_rate=cnn_drop[i],
        )

    # ------------------------------------------------------------------
    # Flatten to 1-D feature vector
    # ------------------------------------------------------------------
    x = tf.keras.layers.Flatten(name="flatten")(x)

    return tf.keras.Model(inputs, x, name=model_name)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_norm_act_drop(x, layer_idx, use_laynorm, use_batchnorm,
                         act_type, drop_rate):
    """Apply normalisation -> activation -> dropout in sequence."""
    if use_laynorm:
        x = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-6, name=f"laynorm_{layer_idx}"
        )(x)
    elif use_batchnorm:
        x = tf.keras.layers.BatchNormalization(
            momentum=0.95, name=f"batchnorm_{layer_idx}"
        )(x)

    x = tf.keras.layers.Activation(
        _act_to_string(act_type), name=f"act_{layer_idx}"
    )(x)
    x = tf.keras.layers.Dropout(drop_rate, name=f"drop_{layer_idx}")(x)
    return x


def _act_to_string(act_type):
    """
    Map SincNet activation names to values accepted by tf.keras.layers.Activation.
    """
    mapping = {
        "relu":       "relu",
        "tanh":       "tanh",
        "sigmoid":    "sigmoid",
        "leaky_relu": tf.keras.layers.LeakyReLU(negative_slope=0.2),
        "elu":        "elu",
        "softmax":    "log_softmax",
        "linear":     tf.keras.layers.LeakyReLU(negative_slope=1.0),
    }
    if act_type not in mapping:
        raise ValueError(f"Unknown activation type: '{act_type}'. "
                         f"Choose from: {list(mapping)}")
    return mapping[act_type]