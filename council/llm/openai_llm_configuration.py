"""

Module Overview:

    This module defines the OpenAILLMConfiguration class which is responsible for managing
    the configuration settings for a connection to the OpenAI API with specific parameters associated
    with the Language Learning Models (LLM).

    The OpenAILLMConfiguration class inherits from LLMConfigurationBase and utilizes environment
    variables for initialization. Parameters such as model, API keys, and API host addresses are handled
    within the class. It also includes validation for several parameters ensuring they meet specific
    criteria. Methods are provided to build a default payload for API requests and to instantiate
    configuration objects from both environment variables and a given LLMConfigSpec specification.

    Classes:
        OpenAILLMConfiguration: Manages and validates the configuration settings for interacting
        with the OpenAI API and its language models.

    Functions:
        OpenAILLMConfiguration.from_env(model=None, api_host=None) -> OpenAILLMConfiguration:
            Creates an OpenAILLMConfiguration instance using the specified model and api_host,
            or falling back on default values sourced from environment variables.

        OpenAILLMConfiguration.from_spec(spec) -> OpenAILLMConfiguration:
            Creates an OpenAILLMConfiguration instance based on a provided LLMConfigSpec object,
            using its parameters and properties to define the configuration.

    Constants:
        _env_var_prefix: A string denoting the prefix used for the OpenAI related environment variables.
        _DEFAULT_TIMEOUT: An integer representing the default timeout value for API requests.


"""
from __future__ import annotations
from typing import Any, Optional

from council.llm import LLMConfigurationBase
from council.llm.llm_config_object import LLMConfigSpec
from council.utils import (
    read_env_str,
    read_env_int,
    Parameter,
    greater_than_validator,
    prefix_validator,
)
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "OPENAI_"


class OpenAILLMConfiguration(LLMConfigurationBase):
    """
    A configuration class for OpenAI LLM integration, which handles API settings.
    This class establishes parameters for interactions with the OpenAI API, including the authentication key,
    the model to use, the API host, and a timeout setting. The class inherits from LLMConfigurationBase,
    providing additional OpenAI-specific functionality and default values.
    
    Parameters:
        api_key (str):
             A string representing the API key required for authenticating with the OpenAI API.
            Must begin with the 'sk-' prefix.
        api_host (str):
             The API host URL for the OpenAI API. Defaults to 'https://api.openai.com'
            if not specified.
        model (str):
             The model identifier to use with the OpenAI API. Must begin with the 'gpt-' prefix.
        timeout (int):
             An optional integer specifying the timeout in seconds for API calls. A positive integer
            is required. Defaults to a pre-defined timeout value if not specified.
    
    Attributes:
        model (Parameter[str]):
             A parameter object representing the model configuration for the API.
        api_key (Parameter[str]):
             A parameter object holding the API key for authentication.
        api_host (Parameter[str]):
             A parameter object containing the API host URL.
        timeout (Parameter[int]):
             A parameter object representing the timeout duration in seconds.
    
    Methods:
        build_default_payload() -> dict[str, Any]:
             Creates a default payload for API requests incorporating
            the model configuration.
        from_env(model:
             Optional[str], api_host: Optional[str]) -> 'OpenAILLMConfiguration': Class method that
            constructs an instance based on environment variable values.
        from_spec(spec:
             LLMConfigSpec) -> 'OpenAILLMConfiguration': Class method for creating an instance from a
            given configuration specification object.
            The class also includes property accessors for the model, api_key, api_host, and timeout attributes.

    """

    def __init__(self, api_key: str, api_host: str, model: str, timeout: int = _DEFAULT_TIMEOUT):
        """
        Initializes a new instance of the client with the specified API parameters.
        This constructor sets up the client with the given API key, API host address, and
        the model to interact with. It also configures a timeout parameter for the API calls.
        
        Args:
            api_key (str):
                 The secret key used to authenticate with the API, beginning with 'sk-'.
            api_host (str):
                 The base URL of the API host, generally beginning with 'http'.
            This parameter is not required and defaults to 'https:
                //api.openai.com'.
            model (str):
                 The identifier of the model to use, usually prefixed with 'gpt-'.
            timeout (int, optional):
                 The maximum amount of time in seconds to wait for an API response.
                This parameter is optional and defaults to the module level
                variable _DEFAULT_TIMEOUT. The timeout must be greater than 0.
        
        Raises:
            ValidationError:
                 If any of the parameters do not pass the validation checks.

        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=prefix_validator("gpt-"))
        self._timeout = Parameter.int(
            name="timeout", required=False, default=timeout, validator=greater_than_validator(0)
        )
        self._api_key = Parameter.string(
            name="api_key", required=True, value=api_key, validator=prefix_validator("sk-")
        )

        self._api_host = Parameter.string(
            name="api_host",
            required=False,
            value=api_host,
            default="https://api.openai.com",
            validator=prefix_validator("http"),
        )

    @property
    def model(self) -> Parameter[str]:
        """
        Gets the model parameter of the instance.
        
        Returns:
            (Parameter[str]):
                 The model parameter of the instance which is a string.

        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        Gets the API key.
        This property method is used to securely access the private `_api_key` attribute from outside
        the class. The API key is expected to be a string that is used for authentication
        and authorization purposes in API calls.
        
        Returns:
            (Parameter[str]):
                 The API key currently set for the instance.

        """
        return self._api_key

    @property
    def api_host(self) -> Parameter[str]:
        """
        Gets the API host URL where requests are sent.
        
        Returns:
            (Parameter[str]):
                 The host URL of the API as a string.

        """
        return self._api_host

    @property
    def timeout(self) -> Parameter[int]:
        """
        Gets the timeout parameter for an object instance.
        
        Returns:
        
        Returns the timeout value which is an integer representing
            the number of units of time before a timeout occurs.

        """
        return self._timeout

    def build_default_payload(self) -> dict[str, Any]:
        """
        Builds the default payload by merging additional data into the payload from the superclass.
        This method constructs a default payload dictionary by first calling the 'build_default_payload' method
        of the superclass to get the initial payload. It then checks if the '_model' attribute contains a value,
        and if it does, the 'model' key is set with the value of '_model' being unwrapped.
        
        Returns:
            (dict[str, Any]):
                 The dictionary containing the default payload. The contents of this payload
                include the superclass payload merged with the 'model' data from this class if '_model'
                is not empty.
            

        """
        payload = super().build_default_payload()
        if self._model.is_some():
            payload.setdefault("model", self._model.unwrap())
        return payload

    @staticmethod
    def from_env(model: Optional[str] = None, api_host: Optional[str] = None) -> OpenAILLMConfiguration:
        """
        Fetches OpenAILLMConfiguration from environment variables.
        This function constructs an instance of OpenAILLMConfiguration using environment variables.
        It fetches the API key, API host, model, and timeout configuration from the environment.
        If certain environment variables are not set, it falls back to default values for API host and model.
        
        Args:
            model (Optional[str]):
                 The model name to use. If None, falls back to the environment variable or default.
            api_host (Optional[str]):
                 The API host URL to use. If None, falls back to the environment variable or default.
        
        Returns:
            (OpenAILLMConfiguration):
                 A configuration object initialized with values gathered from environment variables.
        
        Raises:
            MissingEnvVariableException:
                 If the API_KEY environment variable is required but not set.
            EnvVariableValueException:
                 If an environment variable is expected to be an integer but cannot be converted.

        """
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        if api_host is None:
            api_host = read_env_str(
                _env_var_prefix + "API_HOST", required=False, default="https://api.openai.com"
            ).unwrap()

        if model is None:
            model = read_env_str(_env_var_prefix + "LLM_MODEL", required=False, default="gpt-3.5-turbo").unwrap()

        timeout = read_env_int(_env_var_prefix + "LLM_TIMEOUT", required=False, default=_DEFAULT_TIMEOUT).unwrap()

        config = OpenAILLMConfiguration(model=model, api_key=api_key, api_host=api_host, timeout=timeout)
        config.read_env(_env_var_prefix)
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> OpenAILLMConfiguration:
        """
        Creates an OpenAILLMConfiguration instance from a configuration specification.
        This static method initiates an OpenAILLMConfiguration object with parameters extracted from an LLMConfigSpec object. It requires the 'apiKey' and 'model' to be present within the provided spec. Additional parameters such as 'apiHost' and 'timeout' are optional and have default values if not provided.
        
        Args:
            spec (LLMConfigSpec):
                 An object containing the configuration specifications needed to create an OpenAILLMConfiguration instance. The 'apiKey' and 'model' fields are mandatory, while 'apiHost' and 'timeout' are optional.
        
        Returns:
            (OpenAILLMConfiguration):
                 A fully initialized instance of OpenAILLMConfiguration based on the provided LLMConfigSpec.
        
        Raises:
            MissingConfigurationError:
                 If required configuration parameters like 'apiKey' or 'model' are missing in the spec.

        """
        api_key: str = spec.provider.must_get_value("apiKey")
        api_host: str = spec.provider.get_value("apiHost") or "https://api.openai.com"
        model: str = spec.provider.must_get_value("model")

        config = OpenAILLMConfiguration(api_key=api_key, api_host=api_host, model=str(model))
        if spec.parameters is not None:
            config.from_dict(spec.parameters)

        timeout = spec.provider.get_value("timeout")
        if timeout is not None:
            config.timeout.set(int(timeout))
        return config
