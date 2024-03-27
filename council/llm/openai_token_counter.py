"""

The `openai_token_counter` module provides a class `OpenAITokenCounter` that is responsible for tracking the usage of tokens with OpenAI's machine learning models, particularly in the context of Language Learning Models (LLMs). It offers mechanisms to count tokens within messages, enforce token limits, and filter messages based on token counts. Messages are represented by the `LLMMessage` class, and token overflow is handled through `LLMTokenLimitException`.

Classes:
    OpenAITokenCounter: Inherits from `LLMessageTokenCounterBase` offering methods for token counting and message filtration based on specific encoding schemes and token limits for different OpenAI models.

Raises:
    LLMTokenLimitException: Exception raised when the token count exceeds the specified limit for the model.

Attributes:
    _encoding (Encoding): The `Encoding` object related to the specific OpenAI model being used.
    _model (str): Name of the OpenAI model.
    _limit (int): The maximum number of tokens allowed. Default is -1 indicating no limit.
    _tokens_per_message (int): The baseline token count for each message. Default is 0.
    _tokens_per_name (int): Additional tokens to account for when a name is included in the message. Default is 0.

Methods:
    __init__(self, encoding, model, limit=-1, tokens_per_message=0, tokens_per_name=0): Initializes an instance of OpenAITokenCounter with the specified encoding, model, and token count configurations.
    count_message_token(self, message): Counts and returns the number of tokens in a single LLMMessage.
    count_messages_token(self, messages): Counts and returns the total number of tokens across a sequence of LLMMessage objects and raises `LLMTokenLimitException` if the limit is exceeded.
    filter_first_messages(self, messages, margin): Filters and returns messages from the end until the token limit is reached, considering a margin for additional tokens.
    filter_last_messages(self, messages, margin): Filters and returns messages from the start until the token limit is reached, considering a margin for additional tokens.

   token_limit(self): Property that returns the current token limit for the OpenAITokenCounter instance.

   from_model(model): Factory method that creates and returns an OpenAITokenCounter object configured for a specific OpenAI model based on its token limits and other settings.
    _return_alias(alias, last): Helper static method that returns an OpenAITokenCounter for a given model alias, warning the user that the alias may change over time.



"""
from __future__ import annotations
import logging
import tiktoken

from typing import List, Optional, Sequence
from tiktoken import Encoding

from . import LLMMessage, LLMessageTokenCounterBase, LLMTokenLimitException

logger = logging.getLogger(__name__)


class OpenAITokenCounter(LLMessageTokenCounterBase):
    """
    A counter class to track and manage the token usage of OpenAI language models. The class includes methods to count tokens in individual messages, in a sequence of messages, and to filter messages while adhering to a specified token limit. Additionally, the class can be instantiated based on predefined or custom model parameters, including the specific encoding scheme and token limits for different models.
    
    Attributes:
        _encoding (Encoding):
             The encoding scheme used for computing token count.
        _model (str):
             The model identifier.
        _limit (int):
             The maximum number of tokens allowed.
        _tokens_per_message (int):
             Base number of tokens added per message.
        _tokens_per_name (int):
             Additional tokens added per message if a name is present.
    
    Methods:
        __init__(self, encoding, model, limit, tokens_per_message, tokens_per_name):
             Initializes the token counter with the specified encoding, model, limit, and tokens per message and name.
        count_message_token(self, message):
             Counts the total tokens for a single message.
        count_messages_token(self, messages):
             Sums the token count for all messages in the sequence and checks against the limit.
        filter_first_messages(self, messages, margin):
             Filters messages from the end to meet the token limit, leaving a margin.
        filter_last_messages(self, messages, margin):
             Filters messages from the start to meet the token limit, leaving a margin.
        token_limit(self):
             Returns the token limit.
        from_model(model):
             Creates an instance of the counter based on the model's attributes.
        _return_alias(alias, last):
             Handles aliasing of model names and returns an appropriate token counter instance.
            Each method within the class includes its own detailed explanation, parameters, return types, and potential exceptions.

    """

    def __init__(
        self, encoding: Encoding, model: str, limit: int = -1, tokens_per_message: int = 0, tokens_per_name: int = 0
    ):
        """
        Initializes a new instance with specified parameters for encoding, model, and token limits per message and name.
        
        Args:
            encoding (Encoding):
                 The encoding scheme to be used for processing data.
            model (str):
                 The model identifier or name to be used for computation.
            limit (int, optional):
                 The maximum number of items to process. Defaults to -1, which typically means no limit.
            tokens_per_message (int, optional):
                 The token limit for each message. Defaults to 0, which may indicate that there is no specific token limit per message.
            tokens_per_name (int, optional):
                 The token limit for each name. Defaults to 0, which may indicate that there is no specific token limit per name.
        
        Attributes:
            _encoding (Encoding):
                 Stores the encoding scheme.
            _model (str):
                 Stores the model identifier or name.
            _limit (int):
                 Stores the limit on the number of items to process. If set to -1, it indicates no limit.
            _tokens_per_message (int):
                 Stores the specified token limit for messages.
            _tokens_per_name (int):
                 Stores the specified token limit for names.

        """
        self._encoding = encoding
        self._model = model
        self._limit = limit
        self._tokens_per_message = tokens_per_message
        self._tokens_per_name = tokens_per_name

    def count_message_token(self, message: LLMMessage) -> int:
        """
        Counts the total number of tokens in a given LLMMessage object.
        This function calculates the total number of tokens in a message by first taking a base token count per message and then adding the length of the encoded content and role name from the message. Additionally, if the message has a name, the length of the encoded name and a fixed number of tokens per name are also added to the total.
        
        Args:
            message (LLMMessage):
                 The message object for which the token count is to be calculated.
        
        Returns:
            (int):
                 The total number of tokens for the given message.

        """
        num_tokens = self._tokens_per_message
        num_tokens += len(self._encoding.encode(message.content))
        num_tokens += len(self._encoding.encode(message.role.name))
        if message.name is not None:
            num_tokens += len(self._encoding.encode(message.name))
            num_tokens += self._tokens_per_name
        return num_tokens

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        """
        Counts the total number of tokens in a sequence of LLMMessage instances and validates against a predefined limit.
        This method calculates the cumulative token count of all messages in the input sequence. It also accounts for the
        additional tokens that are added to prime the model for replies. If the computed total exceeds the specified token
        limit of the model, an LLMTokenLimitException is raised, indicating that the token count has gone beyond the
        permissible range for the model.
        
        Args:
            messages:
                 A sequence of LLMMessage instances to compute the token count for.
        
        Returns:
            The total token count of the provided message sequence, including the extra tokens for replies priming.
        
        Raises:
            LLMTokenLimitException:
                 An error indicating that the token count has exceeded the model's limit.

        """
        result = 0
        for message in messages:
            result += self.count_message_token(message)
        result += 3  # every reply is primed with <|start|>assistant<|message|>

        if 0 < self._limit < result:
            raise LLMTokenLimitException(token_count=result, limit=self._limit, model=self._model, llm_name=None)

        return result

    def filter_first_messages(self, messages: Sequence[LLMMessage], margin: int) -> List[LLMMessage]:
        """
        Filters the first messages within a specified token margin from a sequence of messages.
        This function iterates over the provided messages in reverse order, aggregating the count of message tokens and
        includes messages in the result up to a calculated limit which accounts for the specified margin. If the
        aggregated token count plus the token count of the current message does not exceed the limit, the message
        is prepended to the result list. The iteration terminates when the limit is reached, or when all messages
        have been considered.
        
        Args:
            messages (Sequence[LLMMessage]):
                 The sequence of messages to be filtered.
            margin (int):
                 The token margin to consider in the filtering process.
        
        Returns:
            (List[LLMMessage]):
                 A list of messages that meet the criteria of not exceeding the token limit
                when counted in reverse order from the end of the messages sequence.

        """
        count = 0
        limit = self._limit + 3 - margin
        if limit <= 0:
            return []

        result: List[LLMMessage] = []
        for message in reversed(messages):
            token = self.count_message_token(message)
            if count + token < limit:
                count += token
                result.insert(0, message)
            else:
                break
        return result

    def filter_last_messages(self, messages: Sequence[LLMMessage], margin: int) -> List[LLMMessage]:
        """
        Filters out messages from the given sequence, ensuring that the resulting list of messages does not exceed
        a specified token limit (adjusted by a margin).
        The function iterates through the input sequence of messages and accumulatively counts the tokens for each message.
        Messages are added to the result list until the token count is about to exceed the adjusted token limit.
        Messages that would cause the token count to exceed the limit are not included in the result.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of message objects to be filtered.
            margin (int):
                 A value to be subtracted from the token limit to create an adjusted limit.
        
        Returns:
            (List[LLMMessage]):
                 A list of message objects where the accumulated number of tokens does
                not exceed the specified token limit (adjusted by the margin) or an empty list if the adjusted limit is
                less than or equal to zero.
            

        """
        count = 0
        limit = self._limit + 3 - margin
        if limit <= 0:
            return []

        result: List[LLMMessage] = []
        for message in messages:
            token = self.count_message_token(message)
            if count + token < limit:
                count += token
                result.append(message)
            else:
                break
        return result

    @property
    def token_limit(self) -> int:
        """
        Gets the current token limit for an object.
        This property retrieves the current token limit value set for the object. The token limit is an integer value that represents a threshold or boundary for the number of tokens. This might be utilized in contexts where there is a need to impose limits on token generation, token usage, or similar token-related operations to ensure system stability or enforce certain business rules. Access to this property is read-only.
        
        Returns:
            (int):
                 The current token limit value of the object.

        """
        return self._limit

    @staticmethod
    def from_model(model: str) -> Optional[OpenAITokenCounter]:
        """
        Constructs an OpenAITokenCounter based on the given model identifier.
        This static method is responsible for creating an instance of OpenAITokenCounter by mapping
        the provided model name to its corresponding token encoding settings. It handles the
        association of token limits, tokens per message, and tokens per name for various models.
        If the model is unknown, a default encoding is used and a warning is logged. Some model
        names are known aliases and are redirected to their respective current model.
        
        Args:
            model (str):
                 The string identifier of the model for which the token counter is
                to be created.
        
        Returns:
            (Optional[OpenAITokenCounter]):
                 An instance of OpenAITokenCounter configured with
                the appropriate settings for the specified model. Returns None if the provided
                model identifier does not match any known configurations.
        
        Raises:
            KeyError:
                 If the encoding for the provided model cannot be determined and
                a fallback encoding must be used.

        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in {
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
        }:
            tokens_limit = 16384 if ("-16k-" in model) or ("-1106" in model) else 4096
            tokens_per_message = 3
            tokens_per_name = 1
        elif model in {
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
        }:
            tokens_limit = 32768 if "-32k-" in model else 8192
            tokens_per_message = 3
            tokens_per_name = 1
        elif model in {
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
        }:
            tokens_limit = 128000
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_limit = 4096
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            return OpenAITokenCounter._return_alias(model, "gpt-3.5-turbo-0613")
        elif model == "gpt-3.5-turbo-16k":
            return OpenAITokenCounter._return_alias(model, "gpt-3.5-turbo-16k-0613")
        elif model == "gpt-4":
            return OpenAITokenCounter._return_alias(model, "gpt-4-0613")
        elif model == "gpt-4-32k":
            return OpenAITokenCounter._return_alias(model, "gpt-4-32k-0613")
        elif model == "gpt-4-turbo-preview":
            return OpenAITokenCounter._return_alias(model, "gpt-4-0125-preview")
        else:
            return None

        return OpenAITokenCounter(
            encoding,
            model=model,
            limit=tokens_limit,
            tokens_per_message=tokens_per_message,
            tokens_per_name=tokens_per_name,
        )

    @staticmethod
    def _return_alias(alias: str, last: str) -> Optional[OpenAITokenCounter]:
        """
        Logs a warning message indicating that the alias provided may change over time and returns an OpenAITokenCounter instance using the last model provided.
        
        Args:
            alias (str):
                 The model alias provided by the user which could potentially change.
            last (str):
                 The most recent stable model to base the token counter on.
        
        Returns:
            (Optional[OpenAITokenCounter]):
                 An instance of OpenAITokenCounter corresponding to the last model, or None if the operation cannot be completed.

        """
        logger.warning(f"{alias} may change over time. Returning num tokens assuming {last}.")
        return OpenAITokenCounter.from_model(model=last)
