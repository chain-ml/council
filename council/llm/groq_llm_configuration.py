from __future__ import annotations

from typing import Any, Final, Optional

from council.utils import Parameter, not_empty_validator, read_env_str, zero_to_one_validator

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
        self._temperature = Parameter.float(
            name="temperature", required=False, default=0.0, validator=zero_to_one_validator
        )

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
    def temperature(self) -> Parameter[float]:
        """
        Amount of randomness injected into the response.
        Ranges from 0 to 1.
        """
        return self._temperature

    # TODO: more parameters

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
            value: Optional[Any] = spec.parameters.get("temperature", None)
            if value is not None:
                config.temperature.set(float(value))

        return config
