# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import Dict, List, Callable, Any
from allennlp.data.token_indexers.token_indexer import TokenIndexer, \
    TokenType  # Used this as a based class when using BERT + BIDAF in allennlp
from allennlp.data.tokenizers.token import Token
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary

IndexedTokenList = Dict[str, List[Any]]
logger = logging.getLogger(__name__)
# import warnings
# from overrides import overrides


# import pdb
import os
import collections
import unicodedata
import six
import tensorflow as tf
import torch

# Added in modification to use hugging face with thaikeras BERT
from transformers.tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if token.split(): token = token.split()[0]  # to support SentencePiece vocab file
            token = convert_to_unicode(token)
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    if items != '[PAD]' and items != '[CLS]' and items != '[SEP]':  # Added to fix bug in tokenization step
        for item in items:
            # if isinstance(item,list):pdb.set_trace()
            output.append(vocab[item])
        # pdb.set_trace()
    else:
        output.append(vocab[items])
        # pdb.set_trace()
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


from bpe_helper import BPE
import sentencepiece as spm


# Copied from allennlp wordpiece
def _get_token_type_ids(wordpiece_ids: List[int], separator_ids: List[int]) -> List[
    int]:  # Copied from allenlp WordpieceTokenizer
    num_wordpieces = len(wordpiece_ids)
    token_type_ids: List[int] = []
    type_id = 0
    cursor = 0
    while cursor < num_wordpieces:
        # check length
        if num_wordpieces - cursor < len(separator_ids):
            token_type_ids.extend(type_id for _ in range(num_wordpieces - cursor))
            cursor += num_wordpieces - cursor
        # check content
        # when it is a separator
        elif all(
                wordpiece_ids[cursor + index] == separator_id
                for index, separator_id in enumerate(separator_ids)
        ):
            token_type_ids.extend(type_id for _ in separator_ids)
            type_id += 1
            cursor += len(separator_ids)
        # when it is not
        else:
            cursor += 1
            token_type_ids.append(type_id)
    return token_type_ids


class ThaiTokenizer(PreTrainedTokenizer):
    """Tokenizes Thai texts."""

    def __init__(self, vocab_file, spm_file, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", **kwargs):

        # copy the super clause from huggingface repo
        super(ThaiTokenizer, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                            pad_token=pad_token, cls_token=cls_token,
                                            mask_token=mask_token, **kwargs)

        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.bpe = BPE(vocab_file)
        self.s = spm.SentencePieceProcessor()
        self.s.Load(spm_file)
        self.pad_token = pad_token

        # Straigth copied from hugging face repo
        self.max_len = 512  # Hard code 512 for BERT architecture
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        # Added from allennlp wordpiece id
        self._token_min_padding_length = 0  # Copy from token_indexer
        self.has_warned_for_as_padded_tensor = False  # Copy from token_indexer
        self._added_to_vocabulary = False
        namespace = "wordpiece"
        self._namespace = namespace
        self.max_pieces = 512
        self.use_starting_offsets = False
        self._truncate_long_sequences = False  # Set False to use sliding window
        self._warned_about_truncation = False

        separator_token = "[SEP]"
        start_tokens = ["[CLS]"]
        end_tokens = ["[SEP]"]

        # Convert the start_tokens and end_tokens to wordpiece_ids
        self._start_piece_ids = [
            self.vocab[wordpiece]
            for token in (start_tokens or [])
            # for wordpiece in wordpiece_tokenizer(token)
            for wordpiece in self.tokenize(token)
        ]
        self._end_piece_ids = [
            self.vocab[wordpiece]
            for token in (end_tokens or [])
            # for wordpiece in wordpiece_tokenizer(token)
            for wordpiece in self.tokenize(token)
        ]

        # Convert the separator_token to wordpiece_ids
        self._separator_ids = [
            # vocab[wordpiece] for wordpiece in wordpiece_tokenizer(separator_token)
            self.vocab[wordpiece] for wordpiece in self.tokenize(separator_token)
        ]

    def tokenize(self, text, add_special_tokens=None):
        bpe_tokens = self.bpe.encode(text).split(' ')
        spm_tokens = self.s.EncodeAsPieces(text)

        tokens = bpe_tokens if len(bpe_tokens) < len(spm_tokens) else spm_tokens

        split_tokens = []

        for token in tokens:
            new_token = token

            if token.startswith('_') and not token in self.vocab:
                split_tokens.append('_')
                new_token = token[1:]

            if not new_token in self.vocab:
                split_tokens.append('<unk>')
            else:
                split_tokens.append(new_token)

        return split_tokens

    # Added from single_id_token_indexer from allennlp
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, "text_id", None) is None:
            text = token.text
            # if self.lowercase_tokens: # Does not need to lowercase in Thai language
            #     text = text.lower()
            # counter[self.namespace][text] += 1
            counter['bert_token'][text] += 1

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        for word, idx in self.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    def _warn_about_truncation(self, tokens: List[Token]) -> None:
        if not self._warned_about_truncation:
            logger.warning(
                "Too many wordpieces, truncating sequence. "
                "If you would like a sliding window, set `truncate_long_sequences` to False."
                f"The offending input was: {str([token.text for token in tokens])}."
                "To avoid polluting your logs we will not warn about this again."
            )
            self._warned_about_truncation = True

    def get_empty_token_list(self) -> IndexedTokenList:  # Copied from allenlp WordpieceTokenizer
        return {"input_ids": [], "offsets": [], "token_type_ids": [], "mask": []}

    def _add_start_and_end(self, wordpiece_ids: List[int]) -> List[int]:
        return self._start_piece_ids + wordpiece_ids + self._end_piece_ids

    def _extend(self, token_type_ids: List[int]) -> List[int]:
        """
        Extend the token type ids by len(start_piece_ids) on the left
        and len(end_piece_ids) on the right.
        """
        first = token_type_ids[0] if token_type_ids else 0
        last = token_type_ids[-1] if token_type_ids else 0
        return (
                [first for _ in self._start_piece_ids]
                + token_type_ids
                + [last for _ in self._end_piece_ids]
        )

    # Copyied from single_id_token_indexers
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    def get_token_min_padding_length(self) -> int:  # copied from allennlp token_indexer
        """
        This method returns the minimum padding length required for this TokenIndexer.
        For example, the minimum padding length of `SingleIdTokenIndexer` is the largest
        size of filter when using `CnnEncoder`.
        """
        return self._token_min_padding_length

    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}

    def convert_by_vocab_allenlp(self, vocab_allennlp, items):
        # Copied from allennlp wordpiece indexer
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocab_allennlp)
            self._added_to_vocabulary = True

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        # Change from wordpiece_tokenizer to thaibert_tokenizer
        token_wordpiece_ids = [
            [self.vocab[wordpiece] for wordpiece in self.tokenize(token.text)]
            # Add .text after token for thaibert tokenizer
            for token in items
        ]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = _get_token_type_ids(flat_wordpiece_ids, self._separator_ids)

        # The code below will (possibly) pack the wordpiece sequence into multiple sub-sequences by using a sliding
        # window `window_length` that overlaps with previous windows according to the `stride`. Suppose we have
        # the following sentence: "I went to the store to buy some milk". Then a sliding window of length 4 and
        # stride of length 2 will split them up into:

        # "[I went to the] [to the store to] [store to buy some] [buy some milk [PAD]]".

        # This is to ensure that the model has context of as much of the sentence as possible to get accurate
        # embeddings. Finally, the sequences will be padded with any start/end piece ids, e.g.,

        # "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ...".

        # The embedder should then be able to split this token sequence by the window length,
        # pass them through the model, and recombine them.

        # Specify the stride to be half of `self.max_pieces`, minus any additional start/end wordpieces
        window_length = self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        stride = window_length // 2

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = (
            len(self._start_piece_ids)
            if self.use_starting_offsets
            else len(self._start_piece_ids) - 1
        )

        # Count amount of wordpieces accumulated
        pieces_accumulated = 0
        for token in token_wordpiece_ids:
            # Truncate the sequence if specified, which depends on where the offsets are
            next_offset = 1 if self.use_starting_offsets else 0
            if (
                    self._truncate_long_sequences
                    and offset + len(token) - 1 >= window_length + next_offset
            ):
                break

            # For initial offsets, the current value of `offset` is the start of
            # the current wordpiece, so add it to `offsets` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            # For final offsets, the current value of `offset` is the end of
            # the previous wordpiece, so increment it and then add it to `offsets`.
            else:
                offset += len(token)
                offsets.append(offset)

            pieces_accumulated += len(token)

        if len(flat_wordpiece_ids) <= window_length:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)]
            token_type_ids = self._extend(flat_token_type_ids)
        elif self._truncate_long_sequences:
            # self._warn_about_truncation(tokens)
            self._warn_about_truncation(items)
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[:pieces_accumulated])]
            token_type_ids = self._extend(flat_token_type_ids[:pieces_accumulated])
        else:
            # Create a sliding window of wordpieces of length `max_pieces` that advances by `stride` steps and
            # add start/end wordpieces to each window
            # TODO: this currently does not respect word boundaries, so words may be cut in half between windows
            # However, this would increase complexity, as sequences would need to be padded/unpadded in the middle
            wordpiece_windows = [
                self._add_start_and_end(flat_wordpiece_ids[i: i + window_length])
                for i in range(0, len(flat_wordpiece_ids), stride)
            ]

            token_type_windows = [
                self._extend(flat_token_type_ids[i: i + window_length])
                for i in range(0, len(flat_token_type_ids), stride)
            ]

            # Check for overlap in the last window. Throw it away if it is redundant.
            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window):]:
                wordpiece_windows = wordpiece_windows[:-1]
                token_type_windows = token_type_windows[:-1]

            token_type_ids = [token_type for window in token_type_windows for token_type in window]

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]
        return {
            "input_ids": wordpiece_ids,
            "offsets": offsets,
            "token_type_ids": token_type_ids,
            "mask": mask,
        }

    # output = []
    # if items != '[PAD]' and items != '[CLS]' and items != '[SEP]': #Added to fix bug in tokenization step    
    #   for item in items:
    #     pdb.set_trace()
    #     #if isinstance(item,list):pdb.set_trace()
    #     output.append(vocab[item.text]) #add .text to get the string from allennlp.Token
    #   #pdb.set_trace()
    # else:    
    #   output.append(vocab[items])
    #   #pdb.set_trace()
    # return output

    # def convert_tokens_to_ids(self, tokens):
    #   return convert_by_vocab(self.vocab, tokens)
    def tokens_to_indices(self, tokens, vocabulary, indexer_name):  # Added in to use BERT in conjunction with BIDAF
        # Add a dummy vocabulary to make the code works
        # indexer_name is just a dummy argument to make the code work
        return self.convert_by_vocab_allenlp(vocabulary, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    # Added in function from BERT tokenizer in huggingface repo
    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES['vocab_file'])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
