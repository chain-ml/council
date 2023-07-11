from council.utils import read_env


class AzureConfiguration:
    """
    Configuration for :class:AzureLLM

    Args:
        api_key (str): the Azure api key
        api_version (str): the Azure api key. i.e. `2023-03-15-preview`
        api_base (str): the base path for Azure api
        deployment_name (str): the deployment name in Azure
        temperature (int): temperature settings for the LLM
    """

    api_key: str
    api_version: str
    api_base: str
    deployment_name: str
    temperature: int = 0
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "AzureConfiguration":
        config = AzureConfiguration()
        config.api_key = read_env("AZURE_LLM_API_KEY")
        config.api_version = read_env("AZURE_LLM_API_VERSION")
        config.api_base = read_env("AZURE_LLM_API_BASE")
        config.deployment_name = read_env("AZURE_LLM_DEPLOYMENT_NAME")

        timeout = read_env("AZURE_LLM_TIMEOUT", required=False, default="30")
        config.timeout = int(timeout)
        return config
