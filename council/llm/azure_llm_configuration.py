from council.llm import LLMConfigurationBase
from council.utils import read_env_str, read_env_int


class AzureLLMConfiguration(LLMConfigurationBase):
    """
    Configuration for :class:AzureLLM

    Args:
        api_key (str): the Azure api key
        api_version (str): the Azure api key. i.e. `2023-03-15-preview`, `2023-05-15`, `2023-06-01-preview`
        api_base (str): the base path for Azure api
        deployment_name (str): the deployment name in Azure
        temperature (int): temperature settings for the LLM

    Notes:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    """

    api_key: str
    api_version: str
    api_base: str
    deployment_name: str
    timeout: int = 30

    def __init__(self):
        super().__init__("AZURE_")

    @staticmethod
    def from_env() -> "AzureLLMConfiguration":
        config = AzureLLMConfiguration()
        config.read_env()

        config.api_key = read_env_str("AZURE_LLM_API_KEY").unwrap()
        config.api_version = read_env_str("AZURE_LLM_API_VERSION").unwrap()
        config.api_base = read_env_str("AZURE_LLM_API_BASE").unwrap()
        config.deployment_name = read_env_str("AZURE_LLM_DEPLOYMENT_NAME").unwrap()
        config.timeout = read_env_int("AZURE_LLM_TIMEOUT", required=False, default=30).unwrap()
        return config
