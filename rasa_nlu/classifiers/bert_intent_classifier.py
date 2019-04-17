from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import re
import math
import json
import collections

import io
import copy
import logging
import os
from tqdm import tqdm

import typing
from typing import List, Text, Any, Optional, Dict, Tuple

from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
from tensorflow.contrib import predictor
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import TrainingData
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

# --------------------- bert modeling file ------------------------------------
class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
            it is much faster if this is True, on the CPU or GPU, it is faster if
            this is False.
          scope: (optional) variable scope. Defaults to "bert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(
                    self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table

def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
        for TPUs.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(
            flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(
                        attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

# --------------------- bert modeling file ------------------------------------


# --------------------- bert optimization file ------------------------------------
def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

# --------------------- bert optimization file ------------------------------------

class BertIntentClassifier(Component):
    """Intent classifier using google bert.

    """

    name = "intent_classifier_tensorflow_bert"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {

        # bert config
        # The config json file corresponding to the pre-trained BERT model.
        # This specifies the model architecture.
        "bert_config_file": None,
        # The vocabulary file that the BERT model was trained on.
        "vocab_file": None,
        # The output directory where the model checkpoints will be written.
        "output_dir": None,
        # Initial checkpoint (usually from a pre-trained BERT model).
        "init_checkpoint": None,
        # Whether to lower case the input text. Should be True for uncased
        # models and False for cased models.
        "do_lower_case": True,

        # training parameters
        # Total batch size for training.
        "train_batch_size": 32,
        # Total batch size for eval.
        "eval_batch_size": 8,
        # Total batch size for predict.
        "predict_batch_size": 8,
        # The initial learning rate for Adam.
        "learning_rate": 0.00005,
        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter
        # than this will be padded.
        "max_seq_length": 128,
        # number of epochs
        "epochs": 3,

        # Proportion of training to perform linear learning rate warmup for.
        # E.g., 0.1 = 10% of training.
        "warmup_proportion": 0.1,
        # How often to save the model checkpoint.
        "save_checkpoints_steps": 1000,
        # How many steps to make in each estimator call.
        "iterations_per_loop": 1000,

        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000  # large values may hurt performance
    }

    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 inv_intent_dict=None,  # type: Optional[Dict[int, Text]]
                 encoded_all_intents=None,  # type: Optional[np.ndarray]
                 session: Optional['tf.Session'] = None,
                 num_labels=None,
                 predict_fn: Optional['Predictor'] = None,
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""

        self._check_tensorflow()
        super(BertIntentClassifier, self).__init__(component_config)

        self._load_params()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        self.session = session
        self.num_labels = num_labels
        self.predict_fn = predict_fn

        # tf related instances
        self.estimator = None

    # init helpers
    def _load_nn_architecture_params(self, config):
        # type: (Dict[Text, Any]) -> None

        self.vocab_file = config['vocab_file']
        self.output_dir = config['output_dir']
        self.do_lower_case = config['do_lower_case']

        self.train_batch_size = config['train_batch_size']
        self.eval_batch_size = config['eval_batch_size']
        self.predict_batch_size = config['predict_batch_size']

        self.learning_rate = config['learning_rate']
        self.max_seq_length = config['max_seq_length']
        self.epochs = config['epochs']
        self.warmup_proportion = config['warmup_proportion']
        self.init_checkpoint = config['init_checkpoint']
        self.save_checkpoints_steps = config['save_checkpoints_steps']
        self.iterations_per_loop = config['iterations_per_loop']

        self.bert_config = BertConfig.from_json_file(
            config['bert_config_file'])


    def _load_visual_params(self, config):
        # type: (Dict[Text, Any]) -> None
        self.evaluate_every_num_epochs = config['evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config['evaluate_on_num_examples']

    def _load_params(self):
        # type: () -> None

        self._load_nn_architecture_params(self.component_config)
        self._load_visual_params(self.component_config)

    # package safety checks
    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        # type: (TrainingData) -> Dict[Text, int]
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                                for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(intents, intent_split_symbol):
        # type: (List[Text], Text) -> Dict[Text, int]
        """Create intent token dictionary"""

        distinct_tokens = set([token
                               for intent in intents
                               for token in intent.split(
                                   intent_split_symbol)])
        return {token: idx
                for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict):
        # type: (Dict[Text, int]) -> np.ndarray
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """
        return np.eye(len(intent_dict),dtype=int)

    # noinspection PyPep8Naming
    def _create_all_Y(self, size):
        # type: (int) -> np.ndarray
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy
        """

        return np.stack([self.encoded_all_intents] * size)

    # noinspection PyPep8Naming
    def _prepare_data_for_training(self,
                                   training_data,  # type: TrainingData
                                   intent_dict  # type: Dict[Text, int]
                                   ):
        # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        """Prepare data for training"""

        X = np.stack([e.get("text_features")
                      for e in training_data.intent_examples])

        Y = np.array([intent_dict[e.get("intent")]
                                  for e in training_data.intent_examples])

        logger.debug("This information of first example:%s,%s",training_data.intent_examples[0].text,training_data.intent_examples[0].get("tokens"))
        logger.debug("text_features:%s,%s",training_data.intent_examples[0].get("text_features").input_ids,
        training_data.intent_examples[0].get("text_features").input_mask)
        logger.debug("intent[%d]:%s",Y[0],training_data.intent_examples[0].get("intent"))
        # raise Exception("stop!")
        # logger.debug("text_features: X=%s,Y=%s",training_data.intent_examples[0],)
        return X, Y

    # noinspection PyPep8Naming
    def _create_model(self, bert_config, is_training, input_ids, input_mask,
                      labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(
                labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = - \
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    def _input_fn_builder(self, features, labels, seq_length, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to Estimator."""

        all_input_ids = []
        all_input_mask = []
        all_label_ids = labels

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "label_ids":
                    tf.constant(all_label_ids, shape=[
                                num_examples], dtype=tf.int32),
            })

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d

        return input_fn

    def _serving_input_fn_builder(self,max_seq_length):
        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'label_ids': label_ids,
                'input_ids': input_ids,
                'input_mask': input_mask,
            })()
            return input_fn
        return serving_input_fn

    # noinspection PyPep8Naming
    def _model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate,
                          num_train_steps, num_warmup_steps, use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""
        # todo: import optimization
        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            logger.info("*** Features ***")
            for name in sorted(features.keys()):
                logger.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            label_ids = features["label_ids"]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = self._create_model(
                bert_config, is_training, input_ids, input_mask, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint)

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            logger.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                logger.info("  name = %s, shape = %s%s",
                            var.name, var.shape, init_string)

            output_spec = None

            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold=scaffold_fn)

            elif mode == tf.estimator.ModeKeys.EVAL:

                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions)
                loss = tf.metrics.mean(values=per_example_loss)
                eval_metrics = {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=eval_metrics,
                    scaffold=scaffold_fn)
            else:
                # predicted_labels = tf.convert_to_tensor(np.argsort(-probabilities, axis=-1))
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                                "probabilities": probabilities,
                                },
                    scaffold=scaffold_fn)

            return output_spec

        return model_fn

    def _create_estimator(  self,
                            num_lables,
                            num_train_steps = None,
                            num_warmup_steps = None ):

        run_config = tf.estimator.RunConfig(
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps)

        model_fn = self._model_fn_builder(
            bert_config=self.bert_config,
            num_labels=num_lables,
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={
                'batch_size': self.train_batch_size,
            })

        return estimator

    # noinspection PyPep8Naming
    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], Any) -> None
        """Train the bert intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        # self.encoded_all_intents = self._create_encoded_intents(
        #     intent_dict)

        labels_list = sorted(set([example.get("intent")
                                for example in training_data.intent_examples]))
        self.num_labels = len(labels_list)

        logger.debug("num_labels:%d",self.num_labels)

        X, Y = self._prepare_data_for_training(training_data, intent_dict)

        X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.2)

        num_train_steps = int(len(X_train) / self.train_batch_size * self.epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(X_train))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("  Num epochs = %d", self.epochs)

        # self.estimator = self._create_estimator(self.num_labels,num_train_steps,num_warmup_steps)
        
        run_config = tf.estimator.RunConfig(
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps)

        model_fn = self._model_fn_builder(
            bert_config=self.bert_config,
            num_labels=self.num_labels,
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={
                'batch_size': self.train_batch_size,
            })

        train_input_fn = self._input_fn_builder(
            features=X_train,
            labels=Y_train,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_input_fn = self._input_fn_builder(
            features=X_eval,
            labels=Y_eval,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=True)

        tarin_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

        tf.estimator.train_and_evaluate(self.estimator,tarin_spec,eval_spec)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    
    # process helpers
    # noinspection PyPep8Naming
    def process(self, message, **kwargs):
        # type: (Message, Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        # get features (bag of words) for a message
        X = message.get("text_features")
        
        result = self.predict_fn({
            "input_ids": np.array(X.input_ids).reshape(-1, self.max_seq_length),
            "input_mask": np.array(X.input_mask).reshape(-1, self.max_seq_length),
            "label_ids": np.array([]).reshape(-1),
        })

        probabilities = result["probabilities"][0]
        logger.debug("probabilities = %s",probabilities[:10])
        # for (i, prediction) in enumerate(result):
            # logger.debug("i=%d, probabilities = %s",i,prediction["probabilities"][:10])

            # probabilities = prediction["probabilities"]
        intent_ids = np.argsort(-probabilities, axis=-1)
        logger.debug("intent_ids = %s",intent_ids[:10])

        # transform sim to python list for JSON serializing
        probabilities = probabilities.tolist()
        intent_ids = intent_ids.tolist()
        # 
        if len(intent_ids) > 0:
            intent = {"name": self.inv_intent_dict[intent_ids[0]],
                        "confidence": probabilities[intent_ids[0]]}

            # ranking = list(zip(intent_ids, probabilities)
            ranking = intent_ids[:INTENT_RANKING_LENGTH]
            intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                "confidence": probabilities[intent_idx]}
                                for intent_idx in ranking]

            logger.debug("intent_ranking = %s",intent_ranking)

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)


    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        # with io.open(os.path.join(
        #         model_dir,
        #         self.name + "_label_list.pkl"), 'wb') as f:
        #     pickle.dump(self.label_list, f)

        model_path = self.estimator.export_savedmodel(model_dir,
            self._serving_input_fn_builder(self.max_seq_length))

        return {"model_path": model_path.decode('UTF-8'),"num_labels":self.num_labels}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: Any
             ):
        # type: (...) -> EmbeddingIntentClassifier

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("model_path"):

            model_path = os.path.normpath(meta.get("model_path"))

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                predict_fn = predictor.from_saved_model(model_path)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            num_labels = meta.get("num_labels")
            return cls(
                component_config=meta,
                inv_intent_dict=inv_intent_dict,
                encoded_all_intents=encoded_all_intents,
                session=sess,
                num_labels=num_labels,
                predict_fn=predict_fn
            )

            # return cls(
            #         component_config=meta,
            #         inv_intent_dict=inv_intent_dict,
            #         encoded_all_intents=encoded_all_intents,
            # )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
        
    
    
