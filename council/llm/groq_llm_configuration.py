from __future__ import annotations

from typing import Final, List, Mapping, Optional, Tuple, Type

from council.utils import (
    Parameter,
    not_empty_validator,
    penalty_validator,
    positive_validator,
    read_env_str,
    zero_to_one_validator,
    zero_to_two_validator,
)

from . import LLMConfigSpec, LLMConfigurationBase

_env_var_prefix: Final[str] = "GROQ_"


class GroqLLMConfiguration(LLMConfigurationBase):
    def __init__(self, model: str, api_key: str) -> None:
        """
        Initialize a new instance

        Args:
            api_key (str): the api key
            model (str): the model name
        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=not_empty_validator)
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key, validator=not_empty_validator)

        # https://console.groq.com/docs/api-reference#chat
        self._frequency_penalty = Parameter.float(name="frequency_penalty", required=False, validator=penalty_validator)
        self._max_tokens = Parameter.int(name="max_tokens", required=False, validator=positive_validator)
        self._presence_penalty = Parameter.float(name="presence_penalty", required=False, validator=penalty_validator)
        self._seed = Parameter.int(name="seed", required=False)
        self._stop = Parameter.string(name="stop", required=False)
        self._temperature = Parameter.float(
            name="temperature", required=False, default=0.0, validator=zero_to_two_validator
        )
        self._top_p = Parameter.float(name="top_p", required=False, validator=zero_to_one_validator)

    def model_name(self) -> str:
        return self._model.unwrap()

    @property
    def model(self) -> Parameter[str]:
        """
        Groq model name
        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        Groq API Key
        """
        return self._api_key

    @property
    def frequency_penalty(self) -> Parameter[float]:
        """
        Number between -2.0 and 2.0.
        Positive values penalize new tokens based on their existing frequency in the text so far,
        decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self._frequency_penalty

    @property
    def max_tokens(self) -> Parameter[int]:
        """Maximum number of tokens to generate."""
        return self._max_tokens

    @property
    def presence_penalty(self) -> Parameter[float]:
        """
        Number between -2.0 and 2.0.
        Positive values penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.
        """
        return self._presence_penalty

    @property
    def seed(self) -> Parameter[int]:
        """Random seed for generation."""
        return self._seed

    @property
    def stop(self) -> Parameter[str]:
        """Stop sequence."""
        return self._stop

    @property
    def stop_value(self) -> Optional[List[str]]:
        """Format `stop` parameter. Only single value is supported currently."""
        return [self.stop.value] if self.stop.value is not None else None

    @property
    def temperature(self) -> Parameter[float]:
        """
        What sampling temperature to use, between 0 and 2.
        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """Nucleus sampling threshold."""
        return self._top_p

    @staticmethod
    def from_env() -> GroqLLMConfiguration:
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        model = read_env_str(_env_var_prefix + "LLM_MODEL").unwrap()
        config = GroqLLMConfiguration(model=model, api_key=api_key)
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> GroqLLMConfiguration:
        api_key = spec.provider.must_get_value("apiKey")
        model = spec.provider.must_get_value("model")
        config = GroqLLMConfiguration(model=str(model), api_key=str(api_key))

        if spec.parameters is not None:
            param_mapping: Mapping[str, Tuple[Parameter, Type]] = {
                "frequencyPenalty": (config.frequency_penalty, float),
                "maxTokens": (config.max_tokens, int),
                "presencePenalty": (config.presence_penalty, float),
                "seed": (config.seed, int),
                "stop": (config.stop, list),
                "temperature": (config.temperature, float),
                "topP": (config.top_p, float),
            }

            for key, (param, type_conv) in param_mapping.items():
                value = spec.parameters.get(key)
                if value is not None:
                    param.set(type_conv(value))

        return config
