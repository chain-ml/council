"""

Module for integrating and interacting with the Anthropic large language models (LLMs).

This module provides a client for Anthropic's API, an abstract base class
for the Anthropic API client wrapper, and concrete implementations of this wrapper to
handle chat requests. The Anthropic API client wrapper serves as an interface
that must be implemented by subclasses to send chat requests to Anthropic's language models.
The provided implementations transform the messages into the format
required by the Anthropic API's completion and messages endpoints, send the requests,
and handle the respective responses.

Classes:
    AnthropicTokenCounter: A class that extends LLMessageTokenCounterBase to count tokens in messages
        for the consumption calculation according to Anthropic's token counting rules.

    AnthropicLLM: A class that implements the LLMBase abstract class to interface with the
        Anthropic API. This class handles the creation of API client wrappers specific to the
        chosen mode, the posting of chat requests, the conversion of API responses to LLMResult objects,
        error handling, and the computation of consumption statistics.

Functions:
    AnthropicLLM.from_env: Static method to create an AnthropicLLM instance using
        configuration from environment variables.

    AnthropicLLM.from_config: Static method to create an AnthropicLLM instance using
        an LLMConfigObject instance, which must be compatible with the Anthropic provider.



"""
from __future__ import annotations

from typing import Any, Sequence, Optional, List

from anthropic import Anthropic, APITimeoutError, APIStatusError

from council.contexts import LLMContext, Consumption
from council.llm import (
    LLMBase,
    LLMMessage,
    LLMResult,
    LLMCallTimeoutException,
    LLMCallException,
    AnthropicLLMConfiguration,
    LLMessageTokenCounterBase,
    LLMConfigObject,
    LLMProviders,
)
from .anthropic import AnthropicAPIClientWrapper

from .anthropic_completion_llm import AnthropicCompletionLLM
from .anthropic_messages_llm import AnthropicMessagesLLM


class AnthropicTokenCounter(LLMessageTokenCounterBase):
    """
    A class designated for counting tokens in a sequence of messages using an Anthropic client.
    This class inherits from LLMessageTokenCounterBase and utilizes a provided Anthropic client instance to calculate
    the total number of tokens for a given sequence of messages. The total token count is used to quantify the
    amount of data processed, which is critical for systems that have limitations on data throughput.
    
    Attributes:
        _client (Anthropic):
             An instance of the Anthropic class used to count message tokens.
    
    Methods:
        __init__(client:
             Anthropic) -> None:
            Initializes the AnthropicTokenCounter with a given Anthropic client.
    
    Args:
        client (Anthropic):
             An instance of the Anthropic class.
        count_messages_token(messages:
             Sequence[LLMMessage]) -> int:
            Counts the number of tokens in a sequence of LLMMessage instances.
            Iterates through the given sequence of messages, invokes the count_tokens method of the
            Anthropic client to count the number of tokens for each message's content, and then sums
            these values to return the total number of tokens for all messages in the sequence.
    
    Args:
        messages (Sequence[LLMMessage]):
             A sequence of LLMMessage instances whose tokens are to be counted.
    
    Returns:
        (int):
             The total number of tokens in the given sequence of messages.

    """
    def __init__(self, client: Anthropic) -> None:
        """
        Initializes a new instance of the class, setting the client attribute to the provided Anthropic client instance.
        
        Args:
            client (Anthropic):
                 An instance of the Anthropic client which will be used by the class for further interactions.
            

        """
        self._client = client

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        """
        Counts the total number of tokens for a sequence of messages using a client's token counting method.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects whose contents will be tokenized and counted.
        
        Returns:
            (int):
                 The total count of tokens for all messages in the input sequence.

        """
        tokens = 0
        for msg in messages:
            tokens += self._client.count_tokens(msg.content)
        return tokens


class AnthropicLLM(LLMBase):
    """
    A class that manages interactions with Anthropic language model APIs, inheriting from LLMBase.
    This class provides methods to initialize the language model client, post chat requests, convert
    results to consumptions, and create instances from environment configuration or existing config objects.
    It utilizes custom Anthropic API wrapper classes depending on the model configuration.
    
    Attributes:
        config (AnthropicLLMConfiguration):
             The configuration object containing settings such as API key and model specification.
        _client (Anthropic):
             The client interface for Anthropic API.
        _api (AnthropicAPIClientWrapper):
             A wrapper for the Anthropic API to handle specific implementations based on the model.
    
    Methods:
        __init__:
             Class constructor which takes a configuration object and an optional name, initializes the class,
            and creates an Anthropic API client.
        _post_chat_request:
             Submits a chat request to the Anthropic API, handles responses and potential exceptions.
        to_consumptions:
             Calculates token consumption metrics for prompts and responses.
        _get_api_wrapper:
             Determines and returns the appropriate API wrapper based on the model configuration.
        from_env:
             Static method to create an instance of AnthropicLLM from environment configuration.
        from_config:
             Static method to create an instance of AnthropicLLM from a given LLM configuration object.
    
    Raises:
        LLMCallTimeoutException:
             An exception indicating a timeout occurred during an API call.
        LLMCallException:
             A general exception for errors that occur during API calls.
        ValueError:
             An exception indicating that the provider specified in the configuration is incorrect.

    """
    def __init__(self, config: AnthropicLLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initializes a new instance of the class with specified configuration and optional name.
        This constructor method sets up an object by assigning the provided configuration to an instance variable
        and creating an Anthropic client using the API key specified in the configuration. It also invokes
        a method to wrap the API for further interactions.
        
        Args:
            config (AnthropicLLMConfiguration):
                 The configuration object containing necessary parameters like the API key.
            name (Optional[str]):
                 An optional human-readable name for the instance. If not provided, the class name is used.
        
        Attributes:
            config (AnthropicLLMConfiguration):
                 Configuration object passed during instantiation.
            _client (Anthropic):
                 The client for interacting with the Anthropic API.
            _api:
                 The wrapped API object for further interactions with the Anthropic service.

        """
        super().__init__(name=name or f"{self.__class__.__name__}")
        self.config = config
        self._client = Anthropic(api_key=config.api_key.value, max_retries=0)
        self._api = self._get_api_wrapper()

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the LLM API and returns the result.
        This function assembles the texts from the provided messages, sends a chat request to the LLM API,
        and handles the responses or potential exceptions. If the request is successful, it returns an
        'LLMResult' object with all the choices and consumptions. If the API call times out, it raises an
        'LLMCallTimeoutException.' If the API call results in a non-successful status code, it raises
        an 'LLMCallException.'
        
        Args:
            context (LLMContext):
                 The context of the LLM call.
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects to be sent as a chat request.
            **kwargs (Any):
                 Additional keyword arguments that can be passed to the chat request.
        
        Returns:
            (LLMResult):
                 An object containing the results of the chat request including choices and
                consumptions.
        
        Raises:
            LLMCallTimeoutException:
                 If the API call times out.
            LLMCallException:
                 If the API call returns a non-200 status code.
            

        """
        try:
            response = self._api.post_chat_request(messages=messages)
            prompt_text = "\n".join([msg.content for msg in messages])
            return LLMResult(choices=response, consumptions=self.to_consumptions(prompt_text, response))
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self.config.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, prompt: str, responses: List[str]) -> Sequence[Consumption]:
        """
        Calculates the token consumption for a model given a prompt and a list of responses.
        This function takes a prompt and a list of responses, and computes the token consumption for the calling model. Consumption is measured in terms of the number of calls made to the model, the number of tokens in the prompt, the number of tokens in each response, and the total number of tokens used (including both prompt and responses). Each of these consumption metrics is encapsulated in a `Consumption` object, with appropriate labels for identification.
        
        Args:
            prompt (str):
                 The input prompt provided to the model.
            responses (List[str]):
                 The responses output by the model in response to the prompt.
        
        Returns:
            (Sequence[Consumption]):
                 A sequence of `Consumption` objects, representing the various facets of token consumption. The sequence includes consumption for the model call, the prompt tokens, the completion tokens, and the total tokens used.

        """
        model = self.config.model.unwrap()
        prompt_tokens = self._client.count_tokens(prompt)
        completion_tokens = sum(self._client.count_tokens(r) for r in responses)
        return [
            Consumption(1, "call", f"{model}"),
            Consumption(prompt_tokens, "token", f"{model}:prompt_tokens"),
            Consumption(completion_tokens, "token", f"{model}:completion_tokens"),
            Consumption(prompt_tokens + completion_tokens, "token", f"{model}:total_tokens"),
        ]

    def _get_api_wrapper(self) -> AnthropicAPIClientWrapper:
        """
        Defining the method for selecting the appropriate API wrapper instance based on the model configuration.
        Given the model specified in the `self.config`, this method instantiates and returns the right
        `AnthropicAPIClientWrapper` subclass for interacting with the Anthropic API. If the model
        configuration indicates 'claude-2', an instance of `AnthropicCompletionLLM` is returned, otherwise,
        an instance of `AnthropicMessagesLLM` is provided.
        
        Returns:
            (AnthropicAPIClientWrapper):
                 An instance of either `AnthropicCompletionLLM` or
                `AnthropicMessagesLLM`, depending on the model specified in `self.config`.
        
        Raises:
            Exception:
                 If there's an issue with the model configuration, an exception may be raised
                from within the `AnthropicCompletionLLM` or `AnthropicMessagesLLM` initializers.

        """
        if self.config.model.value == "claude-2":
            return AnthropicCompletionLLM(client=self._client, config=self.config)
        return AnthropicMessagesLLM(client=self._client, config=self.config)

    @staticmethod
    def from_env() -> AnthropicLLM:
        """
        Creates a new AnthropicLLM object instance initialized with configuration obtained from environment variables.
        This static method is a factory function for creating a new AnthropicLLM instance, where the configuration is built by reading the current environment variables suitable for initializing an AnthropicLLMConfiguration.
        
        Returns:
            An instance of AnthropicLLM configured with settings derived from environment variables.
        
        Raises:
            LLMConfigError:
                 If there is an error in extracting or parsing the configuration from the environment.

        """

        return AnthropicLLM(AnthropicLLMConfiguration.from_env())

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> AnthropicLLM:
        """
        Creates and returns an instance of AnthropicLLM from a given configuration object.
        This static method constructs an `AnthropicLLM` instance using the settings defined in a `LLMConfigObject`.
        It checks whether the provided LLM provider matches the expected `LLMProviders.Anthropic`, raising a `ValueError` if not.
        Once validated, it uses the configuration from the spec of the configuration object to create the `AnthropicLLM` instance.
        Optionally, a custom name can be specified for the instance through the metadata of the configuration object.
        
        Parameters:
            config_object (LLMConfigObject):
                 The configuration object containing the necessary specifications
                and metadata to construct an instance of `AnthropicLLM`.
        
        Returns:
            (AnthropicLLM):
                 An initialized instance of `AnthropicLLM` configured as per the given `LLMConfigObject`.
        
        Raises:
            ValueError:
                 If the `provider` specified in the configuration object is not of the kind `LLMProviders.Anthropic`.
            

        """
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Anthropic):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Anthropic}")

        config = AnthropicLLMConfiguration.from_spec(config_object.spec)
        return AnthropicLLM(config=config, name=config_object.metadata.name)
