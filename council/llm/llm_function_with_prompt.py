from typing import Any, Iterable, Mapping, Optional, Union

from council.llm import LLMBase, LLMFunction, LLMMessage, LLMMiddlewareChain
from council.llm.llm_function import LLMResponseParser, T_Response
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
    ) -> None:
        """
        Initializes the LLMFunctionWithPrompt with an ability to format system prompt.
        """

        self._prompt_config = prompt_config
        llm_name = llm.configuration.model_name() if isinstance(llm, LLMBase) else llm.llm.configuration.model_name()
        system_prompt = self._prompt_config.get_system_prompt_template(llm_name)
        if system_prompt_params is not None:
            system_prompt = system_prompt.format(**system_prompt_params)

        if not self._prompt_config.has_user_prompt_template:
            raise ValueError("user prompt template is required for LLMFunctionWithPrompt")
        self.user_prompt = self._prompt_config.get_user_prompt_template(llm_name)

        super().__init__(llm, response_parser=response_parser, system_message=system_prompt, max_retries=max_retries)

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

        if user_message is not None or messages is not None:
            raise ValueError(
                "Both `user_message` and `messages` are expected to be None for LLMFunctionWithPrompt.execute "
                "since they are ignored"
            )

        prompt = self.user_prompt.format(**user_prompt_params) if user_prompt_params is not None else self.user_prompt
        return super().execute(user_message=prompt, **kwargs)
