"""

Module for managing Azure Language Learning Models (LLM) configuration.

This module provides a class for configuring and accessing parameters necessary to interact with the Azure LLM service. It includes functionalities to set up configurations using environment variables or by specifying them programmatically. Additionally, the module contains methods to convert configurations to and from dictionaries, as well as to read configuration specifications from a structured dictionary format.

Classes:
    AzureLLMConfiguration: Represents the configuration for an Azure LLM, encapsulating all
        necessary parameters for API interaction.

Functions:
    None

Error Classes:
    None

Constants:
    _env_var_prefix: Defines the prefix for environment variables related to Azure LLM configuration.


"""
from __future__ import annotations
from typing import Optional

from council.llm import LLMConfigurationBase
from council.llm.llm_config_object import LLMConfigSpec
from council.utils import Parameter, read_env_str, greater_than_validator, not_empty_validator
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "AZURE_"


class AzureLLMConfiguration(LLMConfigurationBase):
    """
    Class representing the configuration for Azure Large Language Models.
    This class extends a base configuration class and adds specific parameters required for interfacing with the Azure LLM API. It includes API key, base URL, deployment name, API version, and timeout settings. The class provides methods to initialize the configuration from environment variables or a given specification, as well as to set or get the values of the parameters.
    
    Attributes:
        _api_key (Parameter[str]):
             The parameter holding the API key required for authentication.
        _api_base (Parameter[str]):
             The base URL of the Azure LLM API.
        _deployment_name (Parameter[str]):
             The name of the deployment.
        _api_version (Parameter[str]):
             The version of the Azure LLM API. Defaults to '2023-05-15'.
        _timeout (Parameter[int]):
             The timeout for API requests in seconds. Defaults to a predefined constant value.
    
    Methods:
        __init__:
             Constructor that initializes the AzureLLMConfiguration with the provided API key, base URL, and deployment name.
        from_env:
             A static method that creates an instance of AzureLLMConfiguration from environment variables.
        from_spec:
             A static method that creates an instance of AzureLLMConfiguration from a supplied LLMConfigSpec object.
        api_base:
             Property that returns the base URL Parameter.
        api_key:
             Property that returns the API key Parameter.
        deployment_name:
             Property that returns the deployment name Parameter.
        timeout:
             Property that returns the timeout Parameter.
        api_version:
             Property that returns the API version Parameter.
        _read_optional_env:
             Reads optional environment variables and updates the configuration accordingly.

    """

    def __init__(self, api_key: str, api_base: str, deployment_name: str):
        """
        Initializes a new instance with the necessary API configuration.
        This constructor sets up the various parameters required to interact with an API. It ensures the api_key, api_base, and deployment_name are provided and validated.
        Additionally, it sets default values for api_version and timeout parameters.
        
        Args:
            api_key (str):
                 The API key used for authentication to the API. This must be a non-empty string.
            api_base (str):
                 The base URI of the API. This must be a non-empty string.
            deployment_name (str):
                 The name of the deployment. This must be a non-empty string.
        
        Raises:
            ValueError:
                 If any of `api_key`, `api_base`, or `deployment_name` are empty strings.
            

        """
        super().__init__()
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key, validator=not_empty_validator)
        self._api_base = Parameter.string(name="api_base", required=True, value=api_base, validator=not_empty_validator)
        self._deployment_name = Parameter.string(
            name="deployment_name", required=True, value=deployment_name, validator=not_empty_validator
        )
        self._api_version = Parameter.string(name="api_version", required=False, default="2023-05-15")
        self._timeout = Parameter.int(
            name="timeout", required=False, default=_DEFAULT_TIMEOUT, validator=greater_than_validator(0)
        )

    @property
    def api_base(self) -> Parameter[str]:
        """
        Gets the base API endpoint URL.
        This property method returns the base URL of the API that the client object is currently set to interact with. It is typically used to retrieve the base URL set during initialization of the client object to ensure that requests are transmitted to the correct API endpoint.
        
        Returns:
            (Parameter[str]):
                 A `Parameter` object that wraps around the base API endpoint URL as a string.

        """
        return self._api_base

    @property
    def api_key(self) -> Parameter[str]:
        """
        Gets the API key currently set for the instance.
        
        Returns:
            (Parameter[str]):
                 An object representing the API key.

        """
        return self._api_key

    @property
    def deployment_name(self) -> Parameter[str]:
        """
        Property that retrieves the name of the deployment.
        This property acts as a getter that returns the name of the current deployment as a string.
        
        Returns:
            (Parameter[str]):
                 An object that wraps the deployment name, ensuring it meets certain specifications and constraints as defined by `Parameter` class.
            

        """
        return self._deployment_name

    @property
    def timeout(self) -> Parameter[int]:
        """
        Property that gets the timeout parameter for an operation.
        This property allows retrieval of the timeout setting, which defines
        the maximum time in seconds that an operation can take before it is
        canceled or considered as failed.
        
        Returns:
            (Parameter[int]):
                 An integer representing the timeout duration in seconds.
            

        """
        return self._timeout

    @property
    def api_version(self) -> Parameter[str]:
        """
        Property that retrieves the API version.
        By using this property, one can obtain the version of the API that is currently
        set within the class instance. As it is a property, no arguments are required to
        retrieve the API version. It acts as a getter for the private attribute `_api_version`.
        
        Returns:
            (Parameter[str]):
                 An object of type `Parameter` that encapsulates the API version as a string.
            

        """
        return self._api_version

    def _read_optional_env(self):
        """
        Reads optional environment variables to configure API version and timeout settings for the client.
        This method updates the `api_version` and `_timeout` attributes of the client instance based on the environment variables with the prefixes specified by `_env_var_prefix`. The environment variables looked for are `LLM_API_VERSION` and `LLM_TIMEOUT`, which are concatenated with `_env_var_prefix`.
        Environment variables must be previously set in the environment for them to be considered. If the environment variables are not set, the method will not modify the corresponding attributes.
        
        Attributes updated:
            api_version:
                 An attribute of the client that specifies the API version to be used.
            _timeout:
                 A private attribute that determines the timeout value (in seconds) for API requests.
        
        Raises:
            This method does not explicitly raise any exceptions but may propagate exceptions raised by the `from_env` method of the related attribute objects it attempts to modify.

        """
        self.api_version.from_env(_env_var_prefix + "LLM_API_VERSION")
        self._timeout.from_env(_env_var_prefix + "LLM_TIMEOUT")

    @staticmethod
    def from_env(deployment_name: Optional[str] = None) -> AzureLLMConfiguration:
        """
        Fetches the necessary parameters from environment variables to create an AzureLLMConfiguration object.
        This method looks for certain predefined environment variables that contain
        the configuration required to initialize an Azure LLM service client. If these environment
        variables are not set, the method will throw a MissingEnvVariableException (for required variables)
        or use the provided default values (for optional variables).
        
        Args:
            deployment_name (Optional[str]):
                 The name of the deployment to be used in the configuration.
                If None, the function will try to fetch it from an environment variable. If that also fails
                and the parameter is required, a MissingEnvVariableException will be raised.
        
        Returns:
            (AzureLLMConfiguration):
                 A configured AzureLLMConfiguration instance based on the
                environment variables provided.
        
        Raises:
            MissingEnvVariableException:
                 If the required environment variables are not set
                and no default values are given.

        """
        api_key = read_env_str(_env_var_prefix + "LLM_API_KEY").unwrap()
        api_base = read_env_str(_env_var_prefix + "LLM_API_BASE").unwrap()
        if deployment_name is None:
            deployment_name = read_env_str(_env_var_prefix + "LLM_DEPLOYMENT_NAME", required=False).unwrap()

        config = AzureLLMConfiguration(api_key=api_key, api_base=api_base, deployment_name=deployment_name)
        config.read_env(env_var_prefix=_env_var_prefix)
        config._read_optional_env()
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> AzureLLMConfiguration:
        """
        Create an AzureLLMConfiguration instance from a given LLMConfigSpec.
        This static method instantiates an AzureLLMConfiguration object using the 'apiKey', 'deploymentName',
        and 'apiBase' values from the provided LLMConfigSpec. If the spec also contains parameters, these are
        additionally used to configure the instance. If a 'timeout' is provided within the spec, it is parsed to an
        integer and set accordingly.
        
        Args:
            spec (LLMConfigSpec):
                 A configuration specification object that contains the necessary details
                to construct an AzureLLMConfiguration.
        
        Returns:
            (AzureLLMConfiguration):
                 An instance of AzureLLMConfiguration configured as per the provided
                LLMConfigSpec.
        
        Raises:
            ValueError:
                 If any required value ('apiKey', 'deploymentName', or 'apiBase') is missing from the
                provider within the LLMConfigSpec object.
            TypeError:
                 If the provided timeout value is not an integer or cannot be cast to an integer.

        """
        api_key: str = spec.provider.must_get_value("apiKey")
        deployment_name: str = spec.provider.must_get_value("deploymentName")
        api_base: str = spec.provider.must_get_value("apiBase")
        config = AzureLLMConfiguration(api_key=api_key, api_base=str(api_base), deployment_name=str(deployment_name))

        if spec.parameters is not None:
            config.from_dict(spec.parameters)
        timeout = spec.provider.get_value("timeout")
        if timeout is not None:
            config.timeout.set(int(timeout))
        return config
