from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
import typing
from typing import Any
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from builtins import str


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask):
    
        self.input_ids = input_ids
        self.input_mask = input_mask


class BertFeaturizer(Featurizer):
    name = "intent_featurizer_bert"

    provides = ["text_features"]

    requires = ["tokens"]

    # @classmethod
    # def required_packages(cls):
    #     # type: () -> List[Text]
    #     return ["numpy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        bert_tokenizer = self._bert_tokenizer(**kwargs)
        max_seq_length = kwargs.get('max_seq_length')

        for example in training_data.intent_examples:

            features = self.features_for_tokens(example.get("tokens"),
                                                bert_tokenizer,
                                                max_seq_length)
            example.set("text_features",
                        self._combine_with_existing_text_features(
                            example, features))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        bert_tokenizer = self._bert_tokenizer(**kwargs)
        max_seq_length = kwargs.get('max_seq_length')

        features = self.features_for_tokens(message.get("tokens"),
                                            bert_tokenizer,
                                            max_seq_length)
        message.set("text_features",
                    self._combine_with_existing_text_features(message,
                                                              features))

    def _bert_tokenizer(self, **kwargs):

        bert_tokenizer = kwargs.get("bert_tokenizer")
        if not bert_tokenizer:
            raise Exception("Failed to train 'intent_featurizer_bert'. "
                            "Missing a proper BERT feature extractor. "
                            "Make sure this component is preceded by "
                            "the 'tokenizer_bert' component in the pipeline "
                            "configuration.")
        return bert_tokenizer

    def features_for_tokens(self, tokens, tokenizer, max_seq_length):
        # type: (List[Token]) -> InputFeatures

        tokens_a = tokens
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask)

        return feature
