from __future__ import annotations

from typing import Final, Optional

from council.llm import ChatGPTConfigurationBase
from council.llm.llm_config_object import LLMConfigSpec
from council.utils import Parameter, greater_than_validator, not_empty_validator, read_env_str

_env_var_prefix: Final[str] = "AZURE_"


class AzureChatGPTConfiguration(ChatGPTConfigurationBase):
    """
    Configuration for :class:AzureLLM

    Notes:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    """

    def __init__(self, api_key: str, api_base: str, deployment_name: str, model_name: Optional[str] = None) -> None:
        """
        Initialize a new instance of AzureChatGPTConfiguration
        Args:
            api_key (str): the Azure api key
        """
        super().__init__()
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key, validator=not_empty_validator)
        self._api_base = Parameter.string(name="api_base", required=True, value=api_base, validator=not_empty_validator)
        self._deployment_name = Parameter.string(
            name="deployment_name", required=True, value=deployment_name, validator=not_empty_validator
        )
        self._api_version = Parameter.string(name="api_version", required=False, default="2023-05-15")
        self._timeout = Parameter.int(
            name="timeout", required=False, default=self.default_timeout, validator=greater_than_validator(0)
        )
        self._model_name = model_name or deployment_name

    def model_name(self) -> str:
        return self._model_name

    @property
    def api_base(self) -> Parameter[str]:
        """
        API Base
        """
        return self._api_base

    @property
    def api_key(self) -> Parameter[str]:
        """
        Azure API Key
        """
        return self._api_key

    @property
    def deployment_name(self) -> Parameter[str]:
        """
        Azure deployment name
        """
        return self._deployment_name

    @property
    def timeout(self) -> Parameter[int]:
        """
        API timeout: seconds to wait for response from Azure API before timing out
        """
        return self._timeout

    @property
    def api_version(self) -> Parameter[str]:
        """
        API Version
        The API version to use i.e. `2023-03-15-preview`, `2023-05-15`, `2023-06-01-preview`
        """
        return self._api_version

    def _read_optional_env(self):
        self.api_version.from_env(_env_var_prefix + "LLM_API_VERSION")
        self._timeout.from_env(_env_var_prefix + "LLM_TIMEOUT")

    @staticmethod
    def from_env(deployment_name: Optional[str] = None) -> AzureChatGPTConfiguration:
        api_key = read_env_str(_env_var_prefix + "LLM_API_KEY").unwrap()
        api_base = read_env_str(_env_var_prefix + "LLM_API_BASE").unwrap()
        if deployment_name is None:
            deployment_name = read_env_str(_env_var_prefix + "LLM_DEPLOYMENT_NAME", required=False).unwrap()

        config = AzureChatGPTConfiguration(api_key=api_key, api_base=api_base, deployment_name=deployment_name)
        config.read_env(env_var_prefix=_env_var_prefix)
        config._read_optional_env()
        return config

    @staticmethod
    def from_spec(spec: LLMConfigSpec) -> AzureChatGPTConfiguration:
        api_key: str = spec.provider.must_get_value("apiKey")
        deployment_name: str = spec.provider.must_get_value("deploymentName")
        api_base: str = spec.provider.must_get_value("apiBase")
        config = AzureChatGPTConfiguration(
            api_key=api_key, api_base=str(api_base), deployment_name=str(deployment_name)
        )

        if spec.parameters is not None:
            config.from_dict(spec.parameters)
        timeout = spec.provider.get_value("timeout")
        if timeout is not None:
            config.timeout.set(int(timeout))
        return config
