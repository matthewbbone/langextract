# Copyright 2025 Google LLC.
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

"""Library for breaking documents into Markdown-aware text chunks.

When a text-to-text model (e.g. a large language model with a fixed context
size) can not accommodate a large document, this library can help us break the
document into chunks of a required maximum length that we can perform
inference on.
"""

import bisect
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
from importlib import resources
import functools
import json
import re

from absl import logging
from chonkie import RecursiveChunker
from chonkie import RecursiveRules
import more_itertools

from langextract.core import data
from langextract.core import exceptions
from langextract.core import tokenizer as tokenizer_lib

_MARKDOWN_RECIPE_RESOURCE = "resources/chonkie_markdown_en.json"
_MARKDOWN_RECIPE_NAME = "markdown"
_MARKDOWN_RECIPE_LANGUAGE = "en"
_MARKDOWN_RECIPE_VERSION = "0.1.0"


@functools.lru_cache(maxsize=1)
def _load_markdown_rules() -> RecursiveRules:
  """Loads the pinned Chonkie English Markdown recipe from package data."""
  recipe_path = resources.files("langextract").joinpath(
      _MARKDOWN_RECIPE_RESOURCE
  )
  recipe_data = json.loads(recipe_path.read_text(encoding="utf-8"))
  metadata = recipe_data.get("metadata", {})
  identity = (
      recipe_data.get("name"),
      recipe_data.get("language"),
      metadata.get("version"),
  )
  expected_identity = (
      _MARKDOWN_RECIPE_NAME,
      _MARKDOWN_RECIPE_LANGUAGE,
      _MARKDOWN_RECIPE_VERSION,
  )
  if identity != expected_identity:
    raise ValueError(
        "Unexpected bundled Chonkie recipe identity: "
        f"expected {expected_identity}, got {identity}."
    )
  try:
    rules_data = recipe_data["recipe"]["recursive_rules"]
  except (KeyError, TypeError) as e:
    raise ValueError(
        "Bundled Chonkie Markdown recipe has no recursive rules."
    ) from e
  return RecursiveRules.from_dict(rules_data)


class TokenUtilError(exceptions.LangExtractError):
  """Error raised when token_util returns unexpected values."""


@dataclasses.dataclass
class TextChunk:
  """Stores a text chunk with attributes to the source document.

  Attributes:
    token_interval: The token interval of the chunk in the source document.
    document: The source document.
  """

  token_interval: tokenizer_lib.TokenInterval
  document: data.Document | None = None
  _chunk_text: str | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  _sanitized_chunk_text: str | None = dataclasses.field(
      default=None, init=False, repr=False
  )
  _char_interval: data.CharInterval | None = dataclasses.field(
      default=None, init=False, repr=False
  )

  def __str__(self):
    interval_repr = (
        f"start_index: {self.token_interval.start_index}, end_index:"
        f" {self.token_interval.end_index}"
    )

    doc_id_repr = (
        f"Document ID: {self.document_id}"
        if self.document_id
        else "Document ID: None"
    )

    try:
      chunk_text_repr = f"'{self.chunk_text}'"
    except ValueError:
      chunk_text_repr = "<unavailable: document_text not set>"

    return (
        "TextChunk(\n"
        f"  interval=[{interval_repr}],\n"
        f"  {doc_id_repr},\n"
        f"  Chunk Text: {chunk_text_repr}\n"
        ")"
    )

  @property
  def document_id(self) -> str | None:
    """Gets the document ID from the source document."""
    if self.document is not None:
      return self.document.document_id
    return None

  @property
  def document_text(self) -> tokenizer_lib.TokenizedText | None:
    """Gets the tokenized text from the source document."""
    if self.document is not None:
      return self.document.tokenized_text
    return None

  @property
  def chunk_text(self) -> str:
    """Gets the chunk text. Raises an error if `document_text` is not set."""
    if self.document_text is None:
      raise ValueError("document_text must be set to access chunk_text.")
    if self._chunk_text is None:
      self._chunk_text = get_token_interval_text(
          self.document_text, self.token_interval
      )
    return self._chunk_text

  @property
  def sanitized_chunk_text(self) -> str:
    """Gets the sanitized chunk text."""
    if self._sanitized_chunk_text is None:
      self._sanitized_chunk_text = _sanitize(self.chunk_text)
    return self._sanitized_chunk_text

  @property
  def additional_context(self) -> str | None:
    """Gets the additional context for prompting from the source document."""
    if self.document is not None:
      return self.document.additional_context
    return None

  @property
  def char_interval(self) -> data.CharInterval:
    """Gets the character interval corresponding to the token interval.

    Returns:
      data.CharInterval: The character interval for this chunk.

    Raises:
      ValueError: If document_text is not set.
    """
    if self._char_interval is None:
      if self.document_text is None:
        raise ValueError("document_text must be set to compute char_interval.")
      self._char_interval = get_char_interval(
          self.document_text, self.token_interval
      )
    return self._char_interval


def create_token_interval(
    start_index: int, end_index: int
) -> tokenizer_lib.TokenInterval:
  """Creates a token interval.

  Args:
    start_index: first token's index (inclusive).
    end_index: last token's index + 1 (exclusive).

  Returns:
    Token interval.

  Raises:
    ValueError: If the token indices are invalid.
  """
  if start_index < 0:
    raise ValueError(f"Start index {start_index} must be positive.")
  if start_index >= end_index:
    raise ValueError(
        f"Start index {start_index} must be < end index {end_index}."
    )
  return tokenizer_lib.TokenInterval(
      start_index=start_index, end_index=end_index
  )


def get_token_interval_text(
    tokenized_text: tokenizer_lib.TokenizedText,
    token_interval: tokenizer_lib.TokenInterval,
) -> str:
  """Get the text within an interval of tokens.

  Args:
    tokenized_text: Tokenized documents.
    token_interval: An interval specifying the start (inclusive) and end
      (exclusive) indices of the tokens to extract. These indices refer to the
      positions in the list of tokens within `tokenized_text.tokens`, not the
      value of the field `index` of `token_pb2.Token`. If the tokens are
      [(index:0, text:A), (index:5, text:B), (index:10, text:C)], we should use
      token_interval=[0, 2] to represent taking A and B, not [0, 6]. Please see
      details from the implementation of tokenizer_lib.tokens_text

  Returns:
    Text within the token interval.

  Raises:
    ValueError: If the token indices are invalid.
    TokenUtilError: If tokenizer_lib.tokens_text returns an empty
    string.
  """
  if token_interval.start_index >= token_interval.end_index:
    raise ValueError(
        f"Start index {token_interval.start_index} must be < end index "
        f"{token_interval.end_index}."
    )
  return_string = tokenizer_lib.tokens_text(tokenized_text, token_interval)
  logging.debug(
      "Token util returns string: %s for tokenized_text: %s, token_interval:"
      " %s",
      return_string,
      tokenized_text,
      token_interval,
  )
  if tokenized_text.text and not return_string:
    raise TokenUtilError(
        "Token util returns an empty string unexpectedly. Number of tokens is"
        f" tokenized_text: {len(tokenized_text.tokens)}, token_interval is"
        f" {token_interval.start_index} to {token_interval.end_index}, which"
        " should not lead to empty string."
    )
  return return_string


def get_char_interval(
    tokenized_text: tokenizer_lib.TokenizedText,
    token_interval: tokenizer_lib.TokenInterval,
) -> data.CharInterval:
  """Returns the char interval corresponding to the token interval.

  Args:
    tokenized_text: Document.
    token_interval: Token interval.

  Returns:
    Char interval of the token interval of interest.

  Raises:
    ValueError: If the token_interval is invalid.
  """
  if token_interval.start_index >= token_interval.end_index:
    raise ValueError(
        f"Start index {token_interval.start_index} must be < end index "
        f"{token_interval.end_index}."
    )
  start_token = tokenized_text.tokens[token_interval.start_index]
  # Penultimate token prior to interval.end_index
  final_token = tokenized_text.tokens[token_interval.end_index - 1]
  return data.CharInterval(
      start_pos=start_token.char_interval.start_pos,
      end_pos=final_token.char_interval.end_pos,
  )


def _sanitize(text: str) -> str:
  """Converts all whitespace characters in input text to a single space.

  Args:
    text: Input to sanitize.

  Returns:
    Sanitized text with newlines and excess spaces removed.

  Raises:
    ValueError: If the sanitized text is empty.
  """

  sanitized_text = re.sub(r"\s+", " ", text.strip())
  if not sanitized_text:
    raise ValueError("Sanitized text is empty.")
  return sanitized_text


def make_batches_of_textchunk(
    chunk_iter: Iterator[TextChunk],
    batch_length: int,
) -> Iterable[Sequence[TextChunk]]:
  """Processes chunks into batches of TextChunk for inference, using itertools.batched.

  Args:
    chunk_iter: Iterator of TextChunks.
    batch_length: Number of chunks to include in each batch.

  Yields:
    Batches of TextChunks.
  """
  for batch in more_itertools.batched(chunk_iter, batch_length):
    yield list(batch)


class SentenceIterator:
  """Iterate through sentences of a tokenized text."""

  def __init__(
      self,
      tokenized_text: tokenizer_lib.TokenizedText,
      curr_token_pos: int = 0,
  ):
    """Constructor.

    Args:
      tokenized_text: Document to iterate through.
      curr_token_pos: Iterate through sentences from this token position.

    Raises:
      IndexError: if curr_token_pos is not within the document.
    """
    self.tokenized_text = tokenized_text
    self.token_len = len(tokenized_text.tokens)
    if curr_token_pos < 0:
      raise IndexError(
          f"Current token position {curr_token_pos} can not be negative."
      )
    elif curr_token_pos > self.token_len:
      raise IndexError(
          f"Current token position {curr_token_pos} is past the length of the "
          f"document {self.token_len}."
      )
    self.curr_token_pos = curr_token_pos

  def __iter__(self) -> Iterator[tokenizer_lib.TokenInterval]:
    return self

  def __next__(self) -> tokenizer_lib.TokenInterval:
    """Returns next sentence's interval starting from current token position.

    Returns:
      Next sentence token interval starting from current token position.

    Raises:
      StopIteration: If end of text is reached.
    """
    assert self.curr_token_pos <= self.token_len
    if self.curr_token_pos == self.token_len:
      raise StopIteration
    # This locates the sentence which contains the current token position.
    sentence_range = tokenizer_lib.find_sentence_range(
        self.tokenized_text.text,
        self.tokenized_text.tokens,
        self.curr_token_pos,
    )
    assert sentence_range
    # Start the sentence from the current token position.
    # If we are in the middle of a sentence, we should start from there.
    sentence_range = create_token_interval(
        self.curr_token_pos, sentence_range.end_index
    )
    self.curr_token_pos = sentence_range.end_index
    return sentence_range


class ChunkIterator:
  """Iterates through Markdown-aware chunks produced by Chonkie.

  Chonkie's character offsets are normalized to LangExtract token boundaries
  so downstream alignment continues to use the existing TextChunk contract.
  """

  def __init__(
      self,
      text: str | tokenizer_lib.TokenizedText | None,
      max_char_buffer: int,
      tokenizer_impl: tokenizer_lib.Tokenizer,
      document: data.Document | None = None,
      first_chunk_max_char: int | None = None,
  ):
    """Constructor.

    Args:
      text: Document to chunk. Can be either a string or a tokenized text.
      max_char_buffer: Size of buffer that we can run inference on.
      tokenizer_impl: Tokenizer instance to use.
      document: Optional source document.
      first_chunk_max_char: Optional smaller buffer applied to the first chunk
        only. Used to shift all subsequent chunk boundaries by a fixed amount
        across extraction passes so a span split at a boundary in one pass can
        land whole in another. When None (default), every chunk uses
        max_char_buffer and behavior is unchanged.
    """
    if text is None:
      if document is None:
        raise ValueError("Either text or document must be provided.")
      text = document.text or ""

    if isinstance(text, str):
      text = tokenizer_impl.tokenize(text)
    elif isinstance(text, tokenizer_lib.TokenizedText) and not text.tokens:
      text_to_tokenize = text.text or (document.text if document else "")
      text = tokenizer_impl.tokenize(text_to_tokenize)
    self.tokenized_text = text
    if max_char_buffer <= 0:
      raise ValueError("max_char_buffer must be greater than 0.")
    self.max_char_buffer = max_char_buffer
    self.first_chunk_max_char = first_chunk_max_char

    # TODO: Refactor redundancy between document and text.
    if document is None:
      self.document = data.Document(text=text.text)
    else:
      self.document = document
    self.document.tokenized_text = self.tokenized_text
    self._chunk_iter = iter(self._build_chunks())

  def __iter__(self) -> Iterator[TextChunk]:
    return self

  def __next__(self) -> TextChunk:
    return next(self._chunk_iter)

  @staticmethod
  def _create_recursive_chunker(char_limit: int) -> RecursiveChunker:
    """Creates a character-sized chunker with the pinned Markdown rules."""
    return RecursiveChunker(
        tokenizer="character",
        chunk_size=char_limit,
        rules=_load_markdown_rules(),
        min_characters_per_chunk=1,
    )

  def _candidate_chunk_boundaries(self) -> list[tuple[int, int]]:
    """Returns absolute character ends and limits from Chonkie runs."""
    source_text = self.tokenized_text.text
    if not source_text:
      return []

    first_limit = self.first_chunk_max_char
    if first_limit is None:
      chunks = self._create_recursive_chunker(self.max_char_buffer).chunk(
          source_text
      )
      return [(chunk.end_index, self.max_char_buffer) for chunk in chunks]

    effective_first_limit = max(1, first_limit)
    first_chunks = self._create_recursive_chunker(effective_first_limit).chunk(
        source_text
    )
    if not first_chunks:
      return []

    first_end = first_chunks[0].end_index
    candidate_boundaries = [(first_end, effective_first_limit)]
    if first_end < len(source_text):
      remaining_chunks = self._create_recursive_chunker(
          self.max_char_buffer
      ).chunk(source_text[first_end:])
      candidate_boundaries.extend(
          (first_end + chunk.end_index, self.max_char_buffer)
          for chunk in remaining_chunks
      )
    return candidate_boundaries

  def _token_end_for_char_boundary(
      self,
      char_end: int,
      current_token: int,
      token_starts: Sequence[int],
  ) -> int:
    """Snaps one Chonkie character boundary to a token boundary."""
    tokens = self.tokenized_text.tokens
    token_end = bisect.bisect_left(token_starts, char_end)
    if token_end == 0:
      return 0

    split_token_index = token_end - 1
    split_token = tokens[split_token_index]
    split_interval = split_token.char_interval
    if split_interval.start_pos < char_end < split_interval.end_pos:
      if split_token_index > current_token:
        return split_token_index
      return split_token_index + 1
    return token_end

  def _text_chunks_for_interval(
      self,
      start_token: int,
      end_token: int,
      char_limit: int,
      token_ends: Sequence[int],
  ) -> list[TextChunk]:
    """Creates size-limited TextChunks for one normalized interval."""
    tokens = self.tokenized_text.tokens
    chunks = []
    while start_token < end_token:
      start_char = tokens[start_token].char_interval.start_pos
      interval_end_char = tokens[end_token - 1].char_interval.end_pos
      if interval_end_char - start_char <= char_limit:
        chunk_end = end_token
      else:
        chunk_end = bisect.bisect_right(
            token_ends,
            start_char + char_limit,
            lo=start_token,
            hi=end_token,
        )
        if chunk_end == start_token:
          # Preserve indivisible source tokens even when one exceeds the limit.
          chunk_end += 1
      chunks.append(
          TextChunk(
              token_interval=create_token_interval(start_token, chunk_end),
              document=self.document,
          )
      )
      start_token = chunk_end
    return chunks

  def _build_chunks(self) -> list[TextChunk]:
    """Converts Chonkie character boundaries into LangExtract chunks."""
    tokens = self.tokenized_text.tokens
    if not tokens:
      return []

    token_starts = [token.char_interval.start_pos for token in tokens]
    token_ends = [token.char_interval.end_pos for token in tokens]
    current_token = 0
    chunks = []
    for char_end, char_limit in self._candidate_chunk_boundaries():
      token_end = self._token_end_for_char_boundary(
          char_end, current_token, token_starts
      )
      token_end = min(max(token_end, current_token), len(tokens))
      if token_end == current_token:
        continue
      chunks.extend(
          self._text_chunks_for_interval(
              current_token, token_end, char_limit, token_ends
          )
      )
      current_token = token_end

    if current_token < len(tokens):
      chunks.extend(
          self._text_chunks_for_interval(
              current_token,
              len(tokens),
              self.max_char_buffer,
              token_ends,
          )
      )
    return chunks
