import tensorflow as tf
from dnn_models_tf import *

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

# ---------------------------------------------------------------------------
# Serialisable replacements for Lambda layers
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="sincnet_model")
class ChannelSlice(tf.keras.layers.Layer):
    """
    Extracts a single channel from a (batch, steps, channels) tensor,
    returning (batch, steps, 1). Replaces a Lambda to allow safe serialisation.
    """

    def __init__(self, channel_index, **kwargs):
        super().__init__(**kwargs)
        self.channel_index = channel_index

    def call(self, x):
        c = self.channel_index
        return x[:, :, c:c + 1]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channel_index": self.channel_index})
        return cfg


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
    
def create_sincnet_model(input_shape, sincnet_options, model_name="sincnet_model"):
    """
    Create a SincNet-based feature extraction model using the Keras functional API.

    Supports multi-channel input (e.g. 3-axis IMU) by applying SincConv
    independently to each channel (depthwise), then concatenating the filter
    bank outputs before passing them through the remaining shared Conv1D layers.
    This lets each IMU axis develop its own frequency-domain representation
    while the deeper layers learn cross-axis interactions.

    Reference:
        Ravanelli, M., & Bengio, Y. (2018). Speaker Recognition from raw waveform
        with SincNet. https://arxiv.org/abs/1808.00158

    Architecture (for N input channels):
        Input  (window_size, N)
        -> [Optional] LayerNorm / BatchNorm on input
        -> Split into N single-channel streams
           -> SincConv_fast + |·| + MaxPool1D + [Norm] + Activation + Dropout
        -> Concatenate streams  ->  (steps, N x cnn_N_filt[0])
        -> Conv1D  (repeated for remaining CNN layers)
           -> MaxPool1D + [Norm] + Activation + Dropout
        -> Flatten  ->  1-D feature vector

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample: (window_size, num_channels).
        SincConv is applied independently per channel, so num_channels >= 1.

    sincnet_options : dict
        Configuration dictionary with the following keys:

        cnn_N_filt        : list[int]   - filters per CNN layer.
                                          Layer 0 sets filters *per channel*;
                                          concatenation makes the effective
                                          width N x cnn_N_filt[0] for layer 1+.
        cnn_len_filt      : list[int]   - kernel size per CNN layer
        cnn_max_pool_len  : list[int]   - max-pool size per CNN layer
        cnn_act           : list[str]   - activation per CNN layer
                                          ('relu', 'tanh', 'leaky_relu', ...)
        cnn_drop          : list[float] - dropout rate per CNN layer
        cnn_use_laynorm   : list[bool]  - use LayerNorm after each CNN layer
        cnn_use_batchnorm : list[bool]  - use BatchNorm after each CNN layer
                                          (mutually exclusive with laynorm)
        cnn_use_laynorm_inp   : bool    - apply LayerNorm to the raw input
        cnn_use_batchnorm_inp : bool    - apply BatchNorm to the raw input
        fs                : int         - IMU sample rate (Hz)

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
    ...     "cnn_drop":              [0.0, 0.0, 0.0],
    ...     "cnn_use_laynorm":       [True, True, True],
    ...     "cnn_use_batchnorm":     [False, False, False],
    ...     "cnn_use_laynorm_inp":   True,
    ...     "cnn_use_batchnorm_inp": False,
    ...     "fs":                    100,   # typical IMU sample rate
    ... }
    >>> model = create_sincnet_model(input_shape=(200, 3), sincnet_options=options)
    >>> model.summary()
    """

    window_size, num_channels = input_shape

    cnn_N_filt = sincnet_options["cnn_N_filt"]
    cnn_len_filt = sincnet_options["cnn_len_filt"]
    cnn_max_pool_len = sincnet_options["cnn_max_pool_len"]
    cnn_act = sincnet_options["cnn_act"]
    cnn_drop = sincnet_options["cnn_drop"]
    cnn_use_laynorm = sincnet_options["cnn_use_laynorm"]
    cnn_use_batchnorm = sincnet_options["cnn_use_batchnorm"]
    cnn_use_laynorm_inp = sincnet_options["cnn_use_laynorm_inp"]
    cnn_use_batchnorm_inp = sincnet_options["cnn_use_batchnorm_inp"]
    fs = sincnet_options["fs"]

    N_cnn_lay = len(cnn_N_filt)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs

    # ------------------------------------------------------------------
    # Optional input normalisation
    # Applied before splitting so it is shared across all channels.
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
    # Layer 0 — depthwise SincConv
    #
    # SincConv_fast only accepts single-channel input, so we slice each
    # IMU axis into its own stream, apply an independent SincConv filter
    # bank, and concatenate the results.
    #
    # Result shape after concat: (batch, steps, num_channels x N_filt_0)
    # The subsequent Conv1D layers then learn cross-axis relationships.
    # ------------------------------------------------------------------
    N_filt_0 = int(cnn_N_filt[0])
    len_filt_0 = int(cnn_len_filt[0])
    pool_len_0 = int(cnn_max_pool_len[0])

    channel_outputs = []
    for ch in range(num_channels):
        # Slice out one axis: (batch, window_size) -> (batch, window_size, 1)
        ch_slice = ChannelSlice(ch, name=f"split_ch{ch}")(x)

        # Independent SincConv filter bank for this axis
        ch_sinc = SincConv_fast(
            N_filt_0, len_filt_0, fs,
            padding="valid",
            min_low_hz=0.1,
            min_band_hz=1,
            name=f"sincconv_ch{ch}"
        )(ch_slice)

        # |·| rectification — converts signed filter responses to energy envelopes,
        # matching torch.abs() in the original SincNet forward pass
        ch_sinc = AbsoluteValue(name=f"abs_ch{ch}")(ch_sinc)

        channel_outputs.append(ch_sinc)

    # Merge all per-axis filter banks along the feature axis
    if num_channels > 1:
        x = tf.keras.layers.Concatenate(axis=-1, name="sinc_concat")(channel_outputs)
    else:
        x = channel_outputs[0]

    # Shared pool + norm + activation + dropout for layer 0
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
    # Input to layer 1 has num_channels x N_filt_0 feature maps, so
    # cross-channel interactions are learned from this point onward.
    # ------------------------------------------------------------------
    for i in range(1, N_cnn_lay):
        N_filt = int(cnn_N_filt[i])
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
    # Flatten to 1-D feature vector  (mirrors x.view(batch, -1))
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
    Keras's functional API requires serialisable objects, so layer instances are
    used for activations that have no built-in string alias (leaky_relu, linear).
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

