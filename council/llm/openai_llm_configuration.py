from typing import Any, Optional

from council.llm import LLMConfigurationBase
from council.utils import read_env_str, read_env_int, Option


class OpenAILLMConfiguration(LLMConfigurationBase):
    """
    Configuration for :class:OpenAILLM

    Args:
        api_key (str): the OpenAI api key
        model (str): optional model version to use
        timeout (int): seconds to wait for response from OpenAI before timing out

    Notes:
        * see https://platform.openai.com/docs/api-reference/chat
    """

    api_key: str
    authorization: str  # not a parameter - used to optimize calls
    model: Option[str]
    timeout: int

    def __init__(self, model: Optional[str] = None, timeout: Optional[int] = None, api_key: Optional[str] = None):
        super().__init__()
        self.model = Option(model)
        self.timeout = timeout or 30
        if api_key is not None:
            self._set_api_key(api_key)

    def build_default_payload(self) -> dict[str, Any]:
        payload = super().build_default_payload()
        if self.model.is_some():
            payload.setdefault("model", self.model.unwrap())
        return payload

    @staticmethod
    def from_env(model: Optional[str] = None) -> "OpenAILLMConfiguration":
        config = OpenAILLMConfiguration(model=model)
        config.read_env(env_var_prefix="OPENAI_")

        config._set_api_key(read_env_str("OPENAI_API_KEY").unwrap())
        if config.model.is_none():
            config.model = read_env_str("OPENAI_LLM_MODEL", required=False, default="gpt-3.5-turbo")
        config.timeout = read_env_int("OPENAI_LLM_TIMEOUT", required=False, default=30).unwrap()
        return config

    def _set_api_key(self, key: str) -> None:
        self.api_key = key
        self.authorization = f"Bearer {key}"
