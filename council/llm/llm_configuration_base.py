"""

A module defining the base configuration for the Language Learning Model (LLM).

This module provides an abstract class `LLMConfigurationBase` which encapsulates the
configuration parameters for LLM. It includes parameters like temperature, max tokens,
top_p, n, presence_penalty, and frequency_penalty. The parameters can be initialized through
the environment or by providing a dictionary of values. The class offers property methods to
access the parameters, a method to read them from the environment, a method to build the
default payload, and a method to populate them from a dictionary.

Classes:
    LLMConfigurationBase: An abstract base class for LLM configuration settings.

Functions:
    _tv(x: float): Validator function for temperature value.
    _pv(x: float): Validator function for presence and frequency penalty values.
    _mtv(x: int): Validator function for max tokens value.

Constants:
    _DEFAULT_TIMEOUT: Default timeout for LLM operations, set to 30 seconds.


"""
import abc
from typing import Any, Dict, Optional

from council.utils.parameter import Parameter

_DEFAULT_TIMEOUT = 30


def _tv(x: float):
    """
    Validates that the input number is within the acceptable range [0.0..2.0].
    Checks if the input `x` is a floating-point number falling within the specified
    inclusive range from 0.0 to 2.0. If `x` is outside this range, a ValueError
    is raised with an appropriate error message.
    
    Args:
        x (float):
             The numerical value to validate.
    
    Raises:
        ValueError:
             If `x` is not within the range [0.0..2.0].
        

    """
    if x < 0.0 or x > 2.0:
        raise ValueError("must be in the range [0.0..2.0]")


def _pv(x: float):
    """
    Calculates the present value and ensures it is within a specific range.
    This function checks if the input value `x` is within the accepted range of -2.0 to 2.0.
    If the value is out of this range, the function raises a ValueError.
    
    Args:
        x (float):
             The value for which present value needs to be calculated.
    
    Raises:
        ValueError:
             If `x` is not within the range of [-2.0..2.0].
        

    """
    if x < -2.0 or x > 2.0:
        raise ValueError("must be in the range [-2.0..2.0]")


def _mtv(x: int):
    """
    Validates that a given integer is positive.
    This function checks if the provided integer `x` is strictly greater than zero.
    If `x` is not positive, the function raises a ValueError to indicate that the input is not valid.
    
    Args:
        x (int):
             The integer to be validated.
    
    Raises:
        ValueError:
             An error indicating that `x` must be a positive integer.
        

    """
    if x <= 0:
        raise ValueError("must be positive")


class LLMConfigurationBase(abc.ABC):
    """
    A base class for language model configuration managing different parameters for inference behavior.
    This abstract base class handles the configuration of various parameters including temperature, max tokens, top_p, n (number of responses), presence penalty, and frequency penalty. It provides methods to retrieve and set parameter values, build default payloads, and read configurations from the environment or a dictionary.
    
    Attributes:
        _temperature (Parameter[float]):
             The temperature parameter influencing randomness in the response.
        _max_tokens (Parameter[int]):
             The maximum number of tokens to generate in the response.
        _top_p (Parameter[float]):
             The top_p parameter for controlling the Nucleus sampling.
        _n (Parameter[int]):
             The number of responses to generate.
        _presence_penalty (Parameter[float]):
             A penalty applied for using the same context.
        _frequency_penalty (Parameter[float]):
             A penalty applied for using frequent tokens.
    
    Methods:
        temperature:
             Property providing access to the current temperature setting.
        top_p:
             Property providing access to the current top_p setting.
        max_tokens:
             Property providing access to the current max tokens setting.
        n:
             Property providing access to the current 'n' setting.
        presence_penalty:
             Property providing access to the current presence penalty setting.
        frequency_penalty:
             Property providing access to the current frequency penalty setting.
        read_env(env_var_prefix:
             str): Reads and sets configuration parameters from environment variables.
        build_default_payload():
             Constructs a default JSON payload from the current configuration parameters.
        from_dict(values:
             Dict[str, Any]): Sets configuration parameters from a dictionary, with keys corresponding to parameter names.
            This class is designed to be inherited, and the specific implementation of parameters should be defined by subclasses, including the methods to read from the environment and convert to a dictionary.
        

    """

    def __init__(self):
        """
        Initializes a new instance with various parameters to configure its behavior.
        
        Attributes:
            _temperature (Parameter[float]):
                 A float parameter controlling the temperature,
                which typically affects randomness in the generation process. Is not required and
                defaults to 0.0. It must pass the validator _tv.
            _max_tokens (Parameter[int]):
                 An integer parameter representing the maximum number
                of tokens to generate. This parameter is not required but must be validated
                by _mtv if provided.
            _top_p (Parameter[float]):
                 A float parameter that controls the top p sampling,
                which is a nucleus sampling where the most probable tokens are considered,
                cumulatively making up the probability p. It is an optional parameter.
            _n (Parameter[int]):
                 An integer parameter indicating the number of completions to generate.
                It is not required and defaults to 1.
            _presence_penalty (Parameter[float]):
                 A float parameter that applies a penalty to tokens
                based on their presence in the past context, incentivizing the model to use
                different tokens. It is optional and must pass the validator _pv if provided.
            _frequency_penalty (Parameter[float]):
                 A float parameter that applies a penalty to tokens
                based on their frequency in the past context to prevent the model from repeating itself.
                It is optional and must pass the validator _pv if provided.

        """
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._max_tokens = Parameter.int(name="max_tokens", required=False, validator=_mtv)
        self._top_p = Parameter.float(name="top_p", required=False)
        self._n = Parameter.int(name="n", required=False, default=1)
        self._presence_penalty = Parameter.float(name="presence_penalty", required=False, validator=_pv)
        self._frequency_penalty = Parameter.float(name="frequency_penalty", required=False, validator=_pv)

    @property
    def temperature(self) -> Parameter[float]:
        """
        Property that gets the current temperature value.
        This method is a property decorator that allows the user to access the current
        temperature value stored within a private variable. The temperature is expected
        to be a float value. This property provides a clean and controlled way of retrieving
        the temperature, as opposed to directly accessing the private variable.
        
        Returns:
            (Parameter[float]):
                 The current temperature as a float.
            

        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """
        
        Returns the top_p value for the current instance.
            The top_p value, typically known as nucleus sampling, is a parameter used in probabilistic
            language models during text generation. It defines the cumulative probability threshold for
            choosing the next word, effectively truncating the less probable words out of consideration
            and allowing for more focused and coherent text generation.
        
        Returns:
            (Parameter[float]):
                 A Parameter object holding the top_p value as a floating point,
                representing the cumulative probability threshold used during text
                sampling for generation.

        """
        return self._top_p

    @property
    def max_tokens(self) -> Parameter[int]:
        """
        
        Returns the maximum number of tokens allowed for some process or operation. The actual nature of the tokens

        """
        return self._max_tokens

    @property
    def n(self) -> Parameter[int]:
        """
        Gets the value of the private '_n' attribute representing a parameter of type int.
        
        Returns:
            (Parameter[int]):
                 The value of the private '_n' attribute.

        """
        return self._n

    @property
    def presence_penalty(self) -> Parameter[float]:
        """
        Gets the presence_penalty parameter's value for the object.
        This is a property method that when accessed, returns the
        value of the presence_penalty. 'presence_penalty' influences the
        likelihood of the model repeating the same line verbatim, with higher
        values making duplicates less likely and promoting creativity in
        responses. This property is encapsulated within the object, typically
        accessible through an underscore-prefixed attribute meant for internal
        use.
        
        Returns:
            (Parameter[float]):
                 The current value of the presence_penalty parameter.

        """
        return self._presence_penalty

    @property
    def frequency_penalty(self) -> Parameter[float]:
        """
        
        Returns the frequency penalty parameter used in the model configuration.
            The frequency penalty parameter helps to control how much the model should penalize the new tokens based on their frequency. This is helpful to prevent the model from repeating itself or overusing certain words. This parameter's effect is applied during the text generation process.
        
        Returns:
            (Parameter[float]):
                 A Parameter object that wraps the float value of the frequency penalty.

        """
        return self._frequency_penalty

    def read_env(self, env_var_prefix: str):
        """
        Reads environment variables related to LLM settings and updates corresponding attributes.
        
        Args:
            env_var_prefix (str):
                 The prefix for the environment variables to be read.Series of environment variables with a common prefix
            are expected to define values for the language model parameters. The expected variables are suffixed with:
                '_LLM_TEMPERATURE', '_LLM_MAX_TOKENS', '_LLM_TOP_P', '_LLM_N', '_LLM_PRESENCE_PENALTY', and '_LLM_FREQUENCY_PENALTY'.
        
        Raises:
            ValueError:
                 If any of the environment variables with the expected suffixes cannot be found or
                converted to the respective data types needed for LLM settings.

        """
        self.temperature.from_env(env_var_prefix + "LLM_TEMPERATURE")
        self.max_tokens.from_env(env_var_prefix + "LLM_MAX_TOKENS")
        self.top_p.from_env(env_var_prefix + "LLM_TOP_P")
        self.n.from_env(env_var_prefix + "LLM_N")
        self.presence_penalty.from_env(env_var_prefix + "LLM_PRESENCE_PENALTY")
        self.frequency_penalty.from_env(env_var_prefix + "LLM_FREQUENCY_PENALTY")

    def build_default_payload(self) -> dict[str, Any]:
        """
        Builds a default payload dictionary by adding parameters that have a value.
        This method initializes an empty payload dictionary and then aggregates all
        the parameters that contain some value by utilizing the `add_param` method.
        
        Parameters such as temperature, max_tokens, top_p, n, presence_penalty, and
            frequency_penalty are considered when they have been initialized with some
            value, and if so, are added to the payload with their respective names as keys.
        
        Returns:
            (dict[str, Any]):
                 A dictionary composed of key-value pairs where keys are
                the parameter names and values are the corresponding parameter values,
                for parameters that have been set.

        """
        payload: dict[str, Any] = {}

        def add_param(parameter: Parameter):
            """
            Adds a parameter to a payload if the parameter is set.
            This function checks if the provided `Parameter` object has a value set using `is_some` method. If the value is set, it adds the parameter with its value to a default dictionary `payload` using the parameter's name as the key.
            
            Args:
                parameter (Parameter):
                     The Parameter object to be checked and added to the payload.
            
            Raises:
                AttributeError:
                     If `payload` does not exist or is not accessible in the current scope.
                    Any exceptions raised by `parameter.unwrap()` if the parameter's value retrieval fails.
                

            """
            if parameter.is_some():
                payload.setdefault(parameter.name, parameter.unwrap())

        add_param(self._temperature)
        add_param(self._max_tokens)
        add_param(self._top_p)
        add_param(self._n)
        add_param(self._presence_penalty)
        add_param(self._frequency_penalty)
        return payload

    def from_dict(self, values: Dict[str, Any]):
        """
        Sets the instance attributes from a dictionary of values.
        The method attempts to set various instance attributes based on provided
        dictionary keys. The expected keys are 'temperature', 'n', 'maxTokens', 'topP',
        'presencePenalty', and 'frequencyPenalty'. For each key, if the value is found,
        it converts it to the appropriate type (float or int) and sets the corresponding
        instance attribute using its setter method. If a key is not found, the
        attribute is not altered.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary containing keys corresponding to
                instance attributes. Each key's value should be convertible to the
                required type for that attribute.
        
        Raises:
            ValueError:
                 If the conversion of a dictionary value to the required
                type fails.

        """
        value: Optional[Any] = None
        value = values.get("temperature", None)
        if value is not None:
            self.temperature.set(float(value))
        value = values.get("n", None)
        if value is not None:
            self.n.set(int(value))
        value = values.get("maxTokens", None)
        if value is not None:
            self.max_tokens.set(int(value))
        value = values.get("topP", None)
        if value is not None:
            self.top_p.set(float(value))
        value = values.get("presencePenalty", None)
        if value is not None:
            self.presence_penalty.set(float(value))
        value = values.get("frequencyPenalty", None)
        if value is not None:
            self.frequency_penalty.set(float(value))
