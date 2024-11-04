from __future__ import annotations

from typing import Any, Final, Optional

from council.utils import Parameter, greater_than_validator, read_env_str

from . import LLMConfigSpec, LLMConfigurationBase

_env_var_prefix: Final[str] = "OLLAMA_"


def _tv(x: float) -> None:
    # TODO: is needed?
    """
    Temperature and Top_p Validators
    Sampling temperature to use, between 0. and 1.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("must be in the range [0.0..1.0]")


class OllamaLLMConfiguration(LLMConfigurationBase):
    def __init__(self, model: str) -> None:
        """
        Initialize a new instance

        Args:
            model (str): model name to use from https://ollama.com/library
        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model)
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self._top_k = Parameter.int(name="top_k", required=False, validator=greater_than_validator(0))

    def model_name(self) -> str:
        return self._model.unwrap()

    @property
    def model(self) -> Parameter[str]:
        """
        Ollama model
        """
        return self._model

    @property
    def temperature(self) -> Parameter[float]:
        """
        Amount of randomness injected into the response.
        Ranges from 0 to 1.
        Use temp closer to 0 for analytical / multiple choice, and closer to 1 for creative and generative tasks.
        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """
        Use nucleus sampling.
        In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in
        decreasing probability order and cut it off once it reaches a particular probability specified by top_p.
        """
        return self._top_p

    @property
    def top_k(self) -> Parameter[int]:
        """
        Only sample from the top K options for each subsequent token.
        Used to remove "long tail" low probability responses.
        """
        return self._top_k

    @staticmethod
    def from_env() -> OllamaLLMConfiguration:
        model = read_env_str(_env_var_prefix + "LLM_MODEL").unwrap()
        config = OllamaLLMConfiguration(model=model)
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> OllamaLLMConfiguration:
        model = spec.provider.must_get_value("model")
        config = OllamaLLMConfiguration(model=str(model))

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

        return config
