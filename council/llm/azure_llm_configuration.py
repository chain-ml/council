from typing import Optional

from council.llm import LLMConfigurationBase
from council.utils import read_env_str, read_env_int


class AzureLLMConfiguration(LLMConfigurationBase):
    """
    Configuration for :class:AzureLLM

    Args:
        api_key (str): the Azure api key
        api_version (str): The API version to use i.e. `2023-03-15-preview`, `2023-05-15`, `2023-06-01-preview`
        api_base (str): the base path for Azure api
        deployment_name (str): the deployment name in Azure

    Notes:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    """

    api_key: str
    api_version: str
    api_base: str
    deployment_name: str
    timeout: int

    def __init__(self, api_version: Optional[str] = None, timeout: Optional[int] = None, api_key: Optional[str] = None):
        super().__init__()
        if api_version is not None:
            self.api_version = api_version
        self.timeout = timeout or 30
        if api_key is not None:
            self.api_key = api_key

    @staticmethod
    def from_env() -> "AzureLLMConfiguration":
        config = AzureLLMConfiguration()
        config.read_env(env_var_prefix="AZURE_")

        config.api_key = read_env_str("AZURE_LLM_API_KEY").unwrap()
        config.api_version = read_env_str("AZURE_LLM_API_VERSION").unwrap()
        config.api_base = read_env_str("AZURE_LLM_API_BASE").unwrap()
        config.deployment_name = read_env_str("AZURE_LLM_DEPLOYMENT_NAME").unwrap()
        config.timeout = read_env_int("AZURE_LLM_TIMEOUT", required=False, default=30).unwrap()
        return config
