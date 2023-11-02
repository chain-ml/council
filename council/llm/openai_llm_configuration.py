from typing import Any, Optional

from council.llm import LLMConfigurationBase
from council.utils import read_env_str, read_env_int, Parameter
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "OPENAI_"


def _model_validator(x: str):
    if not x.startswith("gpt-"):
        raise ValueError("must start with `gpt-`")


def _positive_validator(x: int):
    """
    Max Token and Top_K Validator
    Must be positive
    """
    if x <= 0:
        raise ValueError("must be positive")


class OpenAILLMConfiguration(LLMConfigurationBase):
    """
    Configuration for :class:OpenAILLM

    Notes:
        * see https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, api_key: str, model: str, timeout: Optional[int] = None):
        """
        Initialize a new instance of OpenAILLMConfiguration
        Args:
            api_key (str): the OpenAI api key
            model (str): model version to use
            timeout (int): seconds to wait for response from OpenAI before timing out
        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=_model_validator)
        self._timeout = Parameter.int(name="timeout", required=False, default=timeout, validator=_positive_validator)
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key)

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
    def from_env(model: Optional[str] = None) -> "OpenAILLMConfiguration":
        model = (
            "gpt-3.5-turbo" if model is None else read_env_str(_env_var_prefix + "LLM_MODEL", required=False).unwrap()
        )
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        timeout = read_env_int(_env_var_prefix + "LLM_TIMEOUT", required=False, default=_DEFAULT_TIMEOUT).unwrap()
        config = OpenAILLMConfiguration(model=model, api_key=api_key, timeout=timeout)
        config.read_env(_env_var_prefix)
        return config
