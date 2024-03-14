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
    Configuration for :class:OpenAILLM

    Notes:
        * see https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, api_key: str, api_host: str, model: str, timeout: int = _DEFAULT_TIMEOUT):
        """
        Initialize a new instance of OpenAILLMConfiguration
        Args:
            api_key (str): the OpenAI api key
            api_host (str): the OpenAI Host
            model (str): model version to use
            timeout (int): seconds to wait for response from OpenAI before timing out
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
        OpenAI model
        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        OpenAI API Key
        """
        return self._api_key

    @property
    def api_host(self) -> Parameter[str]:
        """
        OpenAI API Host
        """
        return self._api_host

    @property
    def timeout(self) -> Parameter[int]:
        """
        API timeout
        """
        return self._timeout

    def build_default_payload(self) -> dict[str, Any]:
        payload = super().build_default_payload()
        if self._model.is_some():
            payload.setdefault("model", self._model.unwrap())
        return payload

    @staticmethod
    def from_env(model: Optional[str] = None, api_host: Optional[str] = None) -> OpenAILLMConfiguration:
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
