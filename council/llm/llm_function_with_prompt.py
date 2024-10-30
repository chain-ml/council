from typing import Any, Iterable, Mapping, Optional, Union

from council.llm import LLMBase, LLMFunction, LLMFunctionResponse, LLMMessage, LLMMiddlewareChain
from council.llm.llm_function import LLMResponseParser, T_Response
from council.llm.llm_message import LLMCacheControlData
from council.prompt import LLMPromptConfigObject


class LLMFunctionWithPrompt(LLMFunction[T_Response]):
    """
    Represents an LLMFunction created with LLMPrompt
    """

    def __init__(
        self,
        llm: Union[LLMBase, LLMMiddlewareChain],
        response_parser: LLMResponseParser,
        prompt_config: LLMPromptConfigObject,
        max_retries: int = 3,
        system_prompt_params: Optional[Mapping[str, str]] = None,
        system_prompt_caching: bool = False,
    ) -> None:
        """
        Initializes the LLMFunctionWithPrompt with an ability to format and cache system prompt.

        Args:
            system_prompt_params (Optional[Mapping[str, str]]): system prompt params to format system prompt
            system_prompt_caching (bool): whether to cache system prompt (default: False).
                Only Anthropic prompt caching is supported. Note: entire system prompt should be static
        """

        self._prompt_config = prompt_config
        llm_name = llm.configuration.model_name() if isinstance(llm, LLMBase) else llm.llm.configuration.model_name()
        system_prompt = self._prompt_config.get_system_prompt_template(llm_name)
        if system_prompt_params is not None:
            system_prompt = system_prompt.format(**system_prompt_params)

        system_message = LLMMessage.system_message(system_prompt)
        if system_prompt_caching:
            system_message.add_data(LLMCacheControlData.ephemeral())

        if not self._prompt_config.has_user_prompt_template:
            raise ValueError("user prompt template is required for LLMFunctionWithPrompt")
        self.user_prompt = self._prompt_config.get_user_prompt_template(llm_name)

        super().__init__(llm, response_parser=response_parser, system_message=system_message, max_retries=max_retries)

    def execute_with_llm_response(
        self,
        user_message: Optional[Union[str, LLMMessage]] = None,
        messages: Optional[Iterable[LLMMessage]] = None,
        user_prompt_params: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> LLMFunctionResponse[T_Response]:
        """
        Execute LLMFunctionWithPrompt with an ability to format user prompt.
        """

        if user_message is not None or messages is not None:
            raise ValueError(
                "Both `user_message` and `messages` are expected to be None for LLMFunctionWithPrompt.execute "
                "since they are ignored"
            )

        prompt = self.user_prompt.format(**user_prompt_params) if user_prompt_params is not None else self.user_prompt
        return super().execute_with_llm_response(user_message=prompt, **kwargs)

    def execute(
        self,
        user_message: Optional[Union[str, LLMMessage]] = None,
        messages: Optional[Iterable[LLMMessage]] = None,
        user_prompt_params: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> T_Response:
        """
        Execute LLMFunctionWithPrompt with an ability to format user prompt.
        """

        return self.execute_with_llm_response(user_message, messages, user_prompt_params, **kwargs).response
