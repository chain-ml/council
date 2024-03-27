"""

Module for integration with Azure's Large Language Model (LLM) services.

This module provides classes and methods to interact with Azure's Large Language Models
using the `httpx` library for HTTP communication. It includes the ability to configure connections
to Azure LLM, send requests to an Azure LLM for chat completions, handle exceptions raised during
the API call, and more.

Classes:
    AzureOpenAIChatCompletionsModelProvider: Handles the creation and configuration of HTTP
        requests to Azure LLM's chat completions endpoint.
    AzureLLM: A class representing the Azure LLM service, enabling users to interact with chat
        completions functionality and allowing integration with the higher-level OpenAIChatCompletionsModel.

Functions:
    AzureLLM.from_env(deployment_name: Optional[str]) -> AzureLLM:
        Static method that creates an AzureLLM instance using environment variables for configuration.
    AzureLLM.from_config(config_object: LLMConfigObject) -> AzureLLM:
        Static method that creates an AzureLLM instance from a provided LLMConfigObject which contains
        configuration necessary for the connection to Azure LLM.

Exceptions:
    LLMCallTimeoutException: Raised when a request to the LLM service times out.
    LLMCallException: Raised when a request to the LLM service encounters HTTP status errors.


"""
from __future__ import annotations
from typing import Any, Optional

import httpx
from httpx import TimeoutException, HTTPStatusError

from . import OpenAIChatCompletionsModel, LLMCallTimeoutException, LLMCallException
from .azure_llm_configuration import AzureLLMConfiguration
from .llm_config_object import LLMConfigObject, LLMProviders


class AzureOpenAIChatCompletionsModelProvider:
    """
    A provider class that interfaces with the Azure OpenAI deployments for chat completions.
    This class is designed to set up a connection with Azure's OpenAI service for generating
    chat completions using the provided deployment configurations. It supports sending
    POST requests to the API and handling the associated responses and exceptions.
    
    Attributes:
        config (AzureLLMConfiguration):
             An object containing Azure Language Learning Model
            configurations such as API base URL, the name of the deployment, API version,
            API key, and timeout settings.
        _uri (str):
             A formatted URI for making API calls to perform chat completions.
        _name (Optional[str], optional):
             An optional identifier name for the specific
            instance of the model provider.
    
    Methods:
        post_request(payload:
             dict[str, Any]) -> httpx.Response:
            Sends a POST request with the specified payload to the Azure OpenAI chat completion
            API and returns the response.
    
    Args:
        payload (dict[str, Any]):
             The data to include in the request body. Should follow
            the schema expected by the Azure OpenAI API.
    
    Returns:
        (httpx.Response):
             The response object containing the API response data and metadata.
    
    Raises:
        LLMCallTimeoutException:
             If the request times out based on the configured
            timeout duration.
        LLMCallException:
             If an HTTP status error occurs during the API call.
            Note that the Exceptions mentioned are custom exceptions and should be implemented
            in the broader module or package scope.

    """

    def __init__(self, config: AzureLLMConfiguration, name: Optional[str]) -> None:
        """
        Initializes a new instance of a class that interacts with Azure Language Learning Model (LLM) for chat completions.
        This constructor takes in the configuration object for Azure LLM and an optional name, then constructs the endpoint URI for chat completions based on the provided deployment name in the configuration.
        
        Args:
            config (AzureLLMConfiguration):
                 The configuration object containing the necessary parameters to interact with Azure LLM API.
            name (Optional[str]):
                 An optional name for this instance, which can be used for identification or other purposes.
        
        Attributes:
            config (AzureLLMConfiguration):
                 The configuration information required to make API calls.
            _uri (str):
                 The fully constructed URI for accessing the chat completions API endpoint.
            _name (Optional[str]):
                 An optional identifier name for the instance.
        
        Returns:
            (None):
                 This method does not return any values. It's a constructor for initializing the instance state.

        """
        self.config = config
        self._uri = (
            f"{self.config.api_base.value}/openai/deployments/{self.config.deployment_name.value}/chat/completions"
        )
        self._name = name

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Requests a POST operation using the given payload to the server.
        This method sends a POST request to the server using parameters and a payload defined by the instance's configuration. It sets the appropriate headers, including the api-key and Content-Type, as well as query parameters like api-version. The request operation respects a timeout setting. If the request fails due to a timeout, a custom LLMCallTimeoutException is raised. If the server returns an error response, a custom LLMCallException is thrown, which includes the error code and message.
        
        Args:
            payload (dict[str, Any]):
                 The JSON-serializable data to send in the HTTP POST request body.
        
        Returns:
            (httpx.Response):
                 The HTTP response object received from the server upon the POST request.
        
        Raises:
            LLMCallTimeoutException:
                 An exception indicating that the call to the LLM API timed out.
            LLMCallException:
                 An exception for when the LLM API call returns an unexpected HTTP status error.

        """
        headers = {"api-key": self.config.api_key.unwrap(), "Content-Type": "application/json"}
        params = {"api-version": self.config.api_version.value}

        timeout = self.config.timeout.value
        try:
            with httpx.Client(timeout=timeout) as client:
                return client.post(url=self._uri, headers=headers, params=params, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout, self._name) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text, llm_name=self._name) from e


class AzureLLM(OpenAIChatCompletionsModel):
    """
    A class that extends OpenAIChatCompletionsModel to interact with Azure LLM (Large Language Models).
    This class provides methods to create an instance of AzureLLM using either environment-based configuration or a given configuration object.
    It utilizes AzureOpenAIChatCompletionsModelProvider for making POST requests to interact with Azure's LLM services.
    
    Attributes:
        Inherits all attributes from its parent class OpenAIChatCompletionsModel.
    
    Methods:
        __init__(config:
             AzureLLMConfiguration, name: Optional[str]=None):
            Initialize an instance of AzureLLM with a configuration and an optional name.
            If no name provided, it defaults to the class name.
        from_env(deployment_name:
             Optional[str]=None):
            Create an AzureLLM instance configured via environment variables.
            Optionally takes a deployment_name to specify environment configuration scope.
        from_config(config_object:
             LLMConfigObject):
            Create an AzureLLM instance based on a supplied LLMConfigObject.
            Validates if the provider in the config is compatible (i.e., Azure) before instantiation.
    
    Raises:
        ValueError:
             If from_config is provided an LLMConfigObject with an invalid provider for this class.

    """

    def __init__(self, config: AzureLLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initialize a new instance of the configured model provider class.
        This constructor initializes the model provider with a specific configuration for Azure Language Learning Models (LLM).
        If a name is not provided, the name is set to the class name of the object instantiated.
        
        Args:
            config (AzureLLMConfiguration):
                 The configuration for Azure LLM which includes necessary details such as API keys, endpoints, etc.
            name (Optional[str]):
                 The name of the model provider instance. If not provided, it falls back to the class name. Defaults to None.
        
        Raises:
            TypeError:
                 If any of the provided arguments do not conform to the expected types.
            

        """
        name = name or f"{self.__class__.__name__}"
        super().__init__(config, AzureOpenAIChatCompletionsModelProvider(config, name).post_request, None, name)

    @staticmethod
    def from_env(deployment_name: Optional[str] = None) -> AzureLLM:
        """
        Creates an instance of AzureLLM from environment variables.
        This static method retrieves configuration settings from environment variables
        applicable to a particular deployment (optionally specified by `deployment_name`) and
        creates a new AzureLLM instance with these settings.
        
        Args:
            deployment_name (Optional[str]):
                 An optional deployment name to specify which
                environment variables to use for configuration. If `None`, default
                environment variables will be used.
        
        Returns:
            (AzureLLM):
                 A new instance of AzureLLM with the configuration obtained from the
                environment variables.
        
        Raises:
            ValueError:
                 If the environment variables are not properly set or missing.
            

        """
        config: AzureLLMConfiguration = AzureLLMConfiguration.from_env(deployment_name)
        return AzureLLM(config, deployment_name)

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> AzureLLM:
        """
        Creates an AzureLLM instance from a provided LLMConfigObject.
        This static method initializes an AzureLLM instance using the specifications
        provided in a LLMConfigObject. It ensures that the specified provider matches
        the LLMProviders.Azure expected by this class. If the provider does not match,
        an exception is raised. Otherwise, the AzureLLMConfiguration is generated from
        the configuration object specification, and an AzureLLM instance is returned.
        
        Args:
            config_object (LLMConfigObject):
                 The configuration object containing the
                specifications for initializing the AzureLLM instance.
        
        Returns:
            (AzureLLM):
                 An initialized instance of the AzureLLM class.
        
        Raises:
            ValueError:
                 If the provider specified in the configuration object is not of
                the expected kind LLMProviders.Azure.

        """
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Azure):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Azure}")

        config = AzureLLMConfiguration.from_spec(config_object.spec)
        return AzureLLM(config=config, name=config_object.metadata.name)
