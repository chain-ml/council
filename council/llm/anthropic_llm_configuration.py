"""

Module providing configuration for Anthropic Language Model (LLM).

This module contains the `AnthropicLLMConfiguration` class used for configuring
the Anthropic Language Model with specific parameters such as model name,
API key, maximum number of tokens, timeout, temperature, top_p, and top_k.
Provides methods to create configuration instances from environment variables
or from an `LLMConfigSpec` specification.

`AnthropicLLMConfiguration` utilizes `Parameter` objects to hold the configuration
values, each with their own validation rules to ensure proper setup.
It reads environment variables with a specific prefix ('ANTHROPIC_') to allow
setting up configuration externally.

The configuration parameters are:
- `model`: Name of the model, prefixed with 'claude-'.
- `api_key`: API key for accessing the model, prefixed with 'sk-'.
- `max_tokens`: Maximum number of tokens that can be processed.
- `timeout`: Time in seconds before the request times out.
- `temperature`: Controls the randomness of the output (0.0-1.0).
- `top_p`: Nucleus sampling parameter, controlling the mass of probability
  distribution to sample from (0.0-1.0).
- `top_k`: Controls the number of highest probability vocabulary tokens to
  keep for top-k-filtering.

Class Methods:
- `from_env()`: Creates an instance based on the environment variables.
- `from_spec(spec: LLMConfigSpec)`: Generates an instance from a given
  `LLMConfigSpec` specification object.

Class Properties:
- `model`: Accesses the model parameter.
- `api_key`: Accesses the API key parameter.
- `timeout`: Accesses the timeout parameter.
- `temperature`: Accesses the temperature parameter.
- `top_p`: Accesses the top_p parameter.
- `top_k`: Accesses the top_k parameter.
- `max_tokens`: Accesses the max_tokens parameter.

Note:
Some configuration parameters are optional and can be populated using default
values or environment variable settings. Custom validation functions ensure that
each parameter adheres to its expected range and format.


"""
from __future__ import annotations

from typing import Optional, Any

from council.llm import LLMConfigSpec
from council.utils import read_env_str, Parameter, read_env_int, greater_than_validator, prefix_validator
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "ANTHROPIC_"


def _tv(x: float):
    """
    Calculates and returns an adjusted value based on the input float 'x'.
    This method is used to validate that the given float value 'x' falls within the range [0.0, 1.0].
    If 'x' does not fall within the acceptable range, a ValueError exception is raised.
    
    Args:
        x (float):
             The float value to validate for the predefined range.
    
    Raises:
        ValueError:
             If 'x' is not within the range [0.0, 1.0].
    
    Returns:
        None
        

    """
    if x < 0.0 or x > 1.0:
        raise ValueError("must be in the range [0.0..1.0]")


class AnthropicLLMConfiguration:
    """
    A configuration class for Anthropic Language Learning Models (LLMs).
    This class is responsible for managing the configuration parameters required to initialize
    and operate the Anthropic LLMs. It encapsulates several important settings such as
    the model identifier, API key, and token generation limits, while also allowing optional parameters
    for influencing the model's behavior, such as temperature, top_p, and top_k.
    
    Attributes:
        _model (Parameter[str]):
             A compulsory parameter representing the specific model to be used.
        _api_key (Parameter[str]):
             A required parameter to authenticate API requests.
        _max_tokens (Parameter[int]):
             An essential parameter constraining the maximum amount of tokens to generate.
        _timeout (Parameter[int]):
             An optional parameter specifying the maximum time in seconds before timing out.
        _temperature (Parameter[float]):
             An optional parameter determining randomness in response generation. Closer to 0 provides more deterministic outputs.
        _top_p (Parameter[float]):
             An optional parameter to control the nucleus sampling where top p percent of the probability mass is considered for sampling.
        _top_k (Parameter[int]):
             An optional parameter that allows limiting the sample pool to the top k probabilities only.
            The class also includes methods for retrieving settings from environment variables as well as from
            specifications provided in a structured format.
    
    Methods:
        __init__:
             Initializes the configuration using specific parameters for the model, API key, and maximum token limit.
        model:
             Returns the model parameter.
        api_key:
             Returns the API key parameter.
        timeout:
             Returns the timeout parameter.
        temperature:
             Returns the temperature parameter.
        top_p:
             Returns the top_p parameter.
        top_k:
             Returns the top_k parameter.
        max_tokens:
             Returns the max_tokens parameter.
        _read_optional_env:
             Reads optional environment variables and updates the respective parameters.
        from_env:
             Constructs a configuration from environment variables.
        from_spec:
             Constructs a configuration based on a provided LLMConfigSpec instance.

    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
    ):
        """
        Initializes a new instance of the class with the specified model parameters.
        Requires a model name, API key, and the maximum number of tokens that can be generated. Additional optional parameters
        such as timeout, temperature, top_p, and top_k can also be set with default values or validators.
        
        Args:
            model (str):
                 The name of the model, which must start with the prefix 'claude-'.
            api_key (str):
                 The API key to authenticate requests, which must start with the prefix 'sk-'.
            max_tokens (int):
                 The maximum number of tokens to generate.
        
        Attributes:
            _model (Parameter):
                 An instance of Parameter storing the model name with validation.
            _api_key (Parameter):
                 An instance of Parameter storing the request's API key with validation.
            _max_tokens (Parameter):
                 An instance of Parameter holding the max token limit with validation.
            _timeout (Parameter):
                 An optional Parameter for defining the request's timeout with a default value and validation.
            _temperature (Parameter):
                 An optional Parameter for setting generation temperature with a default value and validation.
            _top_p (Parameter):
                 An optional Parameter for setting the nucleus sampling parameter with validation.
            _top_k (Parameter):
                 An optional Parameter for setting the top-k sampling parameter with validation.
        
        Raises:
            ValueError:
                 If the provided values do not pass the specified validators.
        
        Note:
            The 'prefix_validator' function is used to ensure proper prefixes for 'model' and 'api_key'.
            The 'greater_than_validator' function ensures that the numeric parameters are greater than 0.
            The '_tv' represents the validation function for the temperature and top_p where applicable.
            

        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=prefix_validator("claude-"))
        self._api_key = Parameter.string(
            name="api_key", required=True, value=api_key, validator=prefix_validator("sk-")
        )
        self._max_tokens = Parameter.int(
            name="max_tokens", required=True, value=max_tokens, validator=greater_than_validator(0)
        )

        self._timeout = Parameter.int(
            name="timeout", required=False, default=_DEFAULT_TIMEOUT, validator=greater_than_validator(0)
        )
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self._top_k = Parameter.int(name="top_k", required=False, validator=greater_than_validator(0))

    @property
    def model(self) -> Parameter[str]:
        """
        Gets the model attribute.
        
        Returns:
            (Parameter[str]):
                 The model value stored in the instance.

        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        Property that gets the API key.
        This property allows retrieval of the API key which is used to authenticate
        requests in a service or an API. It provides a secure way to manage the key without
        making it directly accessible or modifiable from outside the class that contains it.
        
        Returns:
            (Parameter[str]):
                 An object representing the API key which is kept secure within the class.
            

        """
        return self._api_key

    @property
    def timeout(self) -> Parameter[int]:
        """
        A property that gets the current timeout value.
        This method is used as a getter for the '_timeout' attribute, which is intended to hold an integer
        representing the number of seconds before a timeout occurs.
        
        Returns:
            (int):
                 The current timeout value.
            

        """
        return self._timeout

    @property
    def temperature(self) -> Parameter[float]:
        """
        Retrieves the current temperature value.
        This method acts as a property, allowing one to get the current
        value of the _temperature attribute within an instance of the class.
        
        Returns:
            (Parameter[float]):
                 The current temperature as a floating-point number.
            

        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """
        Gets the top-p parameter value for the sampling strategy.
        This property method returns the current top-p value used to control the 'nucleus sampling' strategy during text generation. Top-p sampling is a stochastic method that selects the next token from the smallest set whose total probability exceeds the parameter p, often improving text quality by avoiding the inclusion of unlikely tokens.
        
        Returns:
            (Parameter[float]):
                 The top-p value for nucleus sampling.

        """
        return self._top_p

    @property
    def top_k(self) -> Parameter[int]:
        """
        Gets the 'top_k' property value representing the number of top elements to retrieve.
        
        Returns:
            (Parameter[int]):
                 An integer representing the number of top elements to retrieve from a collection.

        """
        return self._top_k

    @property
    def max_tokens(self) -> Parameter[int]:
        """
        
        Returns the maximum number of tokens allowed for a particular operation or input sequence.

        """
        return self._max_tokens

    def _read_optional_env(self):
        """
        Reads environment variables to set optional parameters for a language model.
        This method reads the specified environment variables and updates the language model's optional parameters such as temperature, top_p, top_k, and timeout. The environment variables are prefixed with a predefined variable `_env_var_prefix` before the specific parameter names. If the environment variables are set, the corresponding optional parameters in the language model instance are set to these values.
        The method does not return any value and does not take any arguments except for the implicit `self`.
        
        Raises:
            It does not explicitly raise exceptions but any exceptions raised by the `from_env` method of the internal parameters will propagate up if not handled internally.
            

        """
        self._temperature.from_env(_env_var_prefix + "LLM_TEMPERATURE")
        self._top_p.from_env(_env_var_prefix + "LLM_TOP_P")
        self._top_k.from_env(_env_var_prefix + "LLM_TOP_K")
        self._timeout.from_env(_env_var_prefix + "LLM_TIMEOUT")

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        """
        Fetches and constructs an AnthropicLLMConfiguration instance from environment variables.
        This method reads the environment variables specified with the prefix _env_var_prefix and extracts the necessary parameters to instantiate an AnthropicLLMConfiguration object. It primarily reads the API_KEY, LLM_MODEL, and LLM_MAX_TOKENS variables, with LLM_MAX_TOKENS being optional and defaulting to 300 tokens if not set. It then reads additional optional environment variables for temperature, top P, top K, and timeout parameters and integrates them into the configuration object.
        If any of the required environment variables (API_KEY, LLM_MODEL) are missing, a MissingEnvVariableException will be raised.
        If non-optional integer environment variables cannot be converted to integers or if non-optional floating-point environment variables cannot be converted to floats, an EnvVariableValueException will be raised with the appropriate details.
        
        Returns:
            (AnthropicLLMConfiguration):
                 An instance of AnthropicLLMConfiguration with properties configured from the environment variables.
        
        Raises:
            MissingEnvVariableException:
                 If a required environment variable is not found.
            EnvVariableValueException:
                 If an environment variable cannot be converted to the expected type_int or type_float.

        """
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        model = read_env_str(_env_var_prefix + "LLM_MODEL").unwrap()
        max_tokens = read_env_int(_env_var_prefix + "LLM_MAX_TOKENS", required=False, default=300).unwrap()
        config = AnthropicLLMConfiguration(model=model, api_key=api_key, max_tokens=max_tokens)
        config._read_optional_env()
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> AnthropicLLMConfiguration:
        """
        Creates a new instance of `AnthropicLLMConfiguration` populated with values from a `LLMConfigSpec` object.
        This static method extracts necessary parameters from a given `LLMConfigSpec` object to instantiate a new `AnthropicLLMConfiguration`. It fetches the API key, model, and max tokens as mandatory values. Additionally, it searches for optional parameters such as temperature, top_p, top_k, and timeout within the spec, setting them if present. The function guarantees that all required values are retrieved using `must_get_value` and will unwrap default values if the optional parameters are not specified in the spec.
        
        Args:
            spec (LLMConfigSpec):
                 The configuration specification object containing values to initialize the `AnthropicLLMConfiguration`.
        
        Returns:
            (AnthropicLLMConfiguration):
                 A new instance of `AnthropicLLMConfiguration` with values set according to the provided `LLMConfigSpec`.
        
        Raises:
            ValueError:
                 If required values such as the API key, model, or max tokens are missing or invalid in the `LLMConfigSpec`.

        """
        api_key = spec.provider.must_get_value("apiKey")
        model = spec.provider.must_get_value("model")
        max_tokens = spec.provider.must_get_value("maxTokens")
        config = AnthropicLLMConfiguration(model=str(model), api_key=str(api_key), max_tokens=int(max_tokens))

        if spec.parameters is not None:
            value: Optional[Any] = spec.parameters.get("temperature", None)
            if value is not None:
                config.temperature.set(float(value))
            value = spec.parameters.get("topP", None)
            if value is not None:
                config.top_p.set(float(value))
            value = spec.parameters.get("topK", None)
            if value is not None:
                config.top_k.set(int(value))

        timeout = spec.provider.get_value("timeout")
        if timeout is not None:
            config.timeout.set(int(timeout))
        return config
