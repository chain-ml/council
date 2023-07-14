from typing import Any

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
    model: Option[str] = Option.none()
    timeout: int = 30

    def __init__(self):
        super().__init__("OPENAI_")

    def build_default_payload(self) -> dict[str, Any]:
        payload = super().build_default_payload()
        if self.model.is_some():
            payload.setdefault("model", self.model.unwrap())
        return payload

    @staticmethod
    def from_env() -> "OpenAILLMConfiguration":
        config = OpenAILLMConfiguration()
        config.read_env()

        config.api_key = read_env_str("OPENAI_API_KEY").unwrap()
        config.authorization = f"Bearer {config.api_key}"
        config.model = read_env_str("OPENAI_LLM_MODEL", required=False)

        config.timeout = read_env_int("OPENAI_LLM_TIMEOUT", required=False, default=30).unwrap()
        return config
