"""

Module for OpenAI LLM (Language Learning Models) integration.

This module provides classes and methods to interact with OpenAI's language learning models. It enables
the sending of requests and handling of responses to/from OpenAI's chat completion API services.
The key components are configurations based on environment variables or provided specifications, as
well as the OpenAILLM class for constructing a callable language learning model object.

Classes:
    OpenAIChatCompletionsModelProvider: Handles the configuration and making HTTP POST requests to the OpenAI chat completions API.
    OpenAILLM: Wraps the OpenAIChatCompletionsModel to provide an interface for interacting with OpenAI language learning models.

The module provides functionalities such as initializing the model with configuration, sending requests,
building the default payload for the request, and creating an instance from environment variables or
a given configuration object.


"""
from __future__ import annotations
from typing import Any, Optional

import httpx
from httpx import TimeoutException, HTTPStatusError

from . import (
    OpenAIChatCompletionsModel,
    OpenAITokenCounter,
    LLMCallTimeoutException,
    LLMCallException,
)
from .llm_config_object import LLMConfigObject, LLMProviders
from .openai_llm_configuration import OpenAILLMConfiguration


class OpenAIChatCompletionsModelProvider:
    """
    Provides an interface for accessing OpenAI's Chat Completion models.
    This class handles the creation of HTTP requests to the OpenAI API for generating
    chat completions using the specified configuration parameters. It manages authentication,
    request preparation, and response error handling.
    
    Attributes:
        config (OpenAILLMConfiguration):
             Configuration parameters for the LLM API call.
        _headers (dict):
             Headers to be sent in the API call, including authentication tokens.
        _name (Optional[str]):
             An optional name identifying this particular LLM instance.
    
    Methods:
        __init__(self, config:
             OpenAILLMConfiguration, name: Optional[str] = None) -> None:
            Constructs the OpenAIChatCompletionsModelProvider instance with a given configuration
            and an optional name.
        post_request(self, payload:
             dict[str, Any]) -> httpx.Response:
            Sends a POST request to the OpenAI API with the specified payload and returns the response.
            Handles timeout and HTTP status errors by raising custom exceptions.
    
    Raises:
        LLMCallTimeoutException:
             If a timeout occurs during the API call.
        LLMCallException:
             If the HTTP request fails with a status error.

    """

    def __init__(self, config: OpenAILLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initializes a new instance of a class responsible for handling interactions with an OpenAI Language Model (LLM) API.
        
        Args:
            config (OpenAILLMConfiguration):
                 The configuration object containing relevant details like the API key.
            name (Optional[str], optional):
                 An optional name that could be used to identify the instance. Defaults to None.
                Attaches the passed configuration object to the instance, constructs the appropriate authorization headers for API interaction, and optionally sets a name for the instance.

        """
        self.config = config
        bearer = f"Bearer {config.api_key.unwrap()}"
        self._headers = {"Authorization": bearer, "Content-Type": "application/json"}
        self._name = name

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Post a request with a given payload to a preconfigured URI.
        This method sends a POST request to the URI derived from the API host and path specified in the configuration. It includes the necessary headers and the JSON-serialized payload. It captures timeouts and HTTP status errors, raising custom exceptions to handle these cases.
        
        Args:
            payload (dict[str, Any]):
                 A dictionary representing the JSON payload to be sent in the post request.
        
        Returns:
            (httpx.Response):
                 The HTTP response object returned by the httpx client.
        
        Raises:
            LLMCallTimeoutException:
                 If the request times out based on the pre-set timeout in the configuration.
            LLMCallException:
                 If the response returned by the server indicates a problematic HTTP status.

        """
        uri = self.config.api_host.unwrap() + "/v1/chat/completions"

        timeout = self.config.timeout.unwrap()
        try:
            with httpx.Client(timeout=timeout) as client:
                return client.post(url=uri, headers=self._headers, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout=timeout, llm_name=self._name) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text, llm_name=self._name) from e


class OpenAILLM(OpenAIChatCompletionsModel):
    """
    A class representing the OpenAI large language model (LLM) interface as a subclass of OpenAIChatCompletionsModel, capable of handling configurations and instantiation from environment variables or configuration objects.
    This class offers methods to create instances of the OpenAI language model either from environment configurations using the `from_env` class method or from a specific configuration object through the `from_config` class method. It initializes the LLM model by accepting a configuration, optionally a name, and employing the counter for model tokens.
    
    Attributes:
        None defined.
    
    Methods:
        __init__(self, config:
             OpenAILLMConfiguration, name: Optional[str]=None):
            Initializes a new instance of the OpenAILLM class, setting the model's configuration and optionally its name.
        from_env(cls, model:
             Optional[str]=None, api_host: Optional[str]=None) -> OpenAILLM:
            A class method that creates an instance of OpenAILLM using environment configurations.
        from_config(cls, config_object:
             LLMConfigObject) -> OpenAILLM:
            A class method that creates an instance of OpenAILLM from a LLMConfigObject, verifying the provider matches OpenAI requirements.
    
    Raises:
        ValueError:
             If the provider specified in the configuration object is not OpenAI.

    """

    def __init__(self, config: OpenAILLMConfiguration, name: Optional[str] = None):
        """
        Initialize the class with a given configuration and an optional name.
        This method sets up the class by taking a configuration object of type `OpenAILLMConfiguration`,
        initializing model providers, configuring the post request method, and setting up a token counter
        based on the provided model configuration. If no name is provided, it generates a name by using the class name.
        
        Args:
            config (OpenAILLMConfiguration):
                 An instance of `OpenAILLMConfiguration` which contains the necessary
                configurations for the large language model.
            name (Optional[str], optional):
                 An optional string that represents the name of the instance. If not
                provided, the class name will be used. Defaults to None.
            

        """
        name = name or f"{self.__class__.__name__}"
        super().__init__(
            config,
            OpenAIChatCompletionsModelProvider(config, name).post_request,
            token_counter=OpenAITokenCounter.from_model(config.model.unwrap_or("")),
            name=name,
        )

    @staticmethod
    def from_env(model: Optional[str] = None, api_host: Optional[str] = None) -> OpenAILLM:
        """
        Creates an OpenAILLM instance from environment variables.
        This static method initializes an OpenAILLMConfiguration object using the specified model and
        API host, with fallback to environment variables, and creates an OpenAILLM object with
        the configured parameters.
        
        Args:
            model (Optional[str]):
                 The specific model to be used. If None, the method will
                try to use the model specified in the environment variables.
            api_host (Optional[str]):
                 The specific API host to be used. If None, the method
                will try to use the API host specified in the environment variables.
        
        Returns:
            (OpenAILLM):
                 An instance of OpenAILLM initialized with the specified configuration,
                or configurations retrieved from environment variables if not provided.
        
        Raises:
            ValueError:
                 If the required environment variables are not set, or if they
                contain invalid values, potentially resulting in an invalid configuration.

        """
        config: OpenAILLMConfiguration = OpenAILLMConfiguration.from_env(model=model, api_host=api_host)
        return OpenAILLM(config)

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> OpenAILLM:
        """
        Creates an OpenAILLM instance using a given LLM configuration object.
        Given an LLM configuration object, this function verifies if the provider type matches
        OpenAI. If it does, it further constructs an OpenAILLMConfiguration from the specification
        inside the object, and finally creates an OpenAILLM instance with the obtained configuration
        and optional name from the metadata.
        
        Args:
            config_object (LLMConfigObject):
                 The configuration object that must be of the LLMProviders.OpenAI kind.
        
        Returns:
            (OpenAILLM):
                 An instance of OpenAILLM initialized with the given configuration.
        
        Raises:
            ValueError:
                 If the provider in the config object is not of kind LLMProviders.OpenAI.
            

        """
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.OpenAI):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.OpenAI}")
        config = OpenAILLMConfiguration.from_spec(config_object.spec)
        return OpenAILLM(config=config, name=config_object.metadata.name)
