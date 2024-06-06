import time
from typing import Any, Sequence

from council.contexts import LLMContext
from council.llm import LLMBase, LLMCallException, LLMConfigurationBase, LLMException, LLMMessage, LLMResult
from council.llm.llm_base import T_Configuration


class LLMFallbackConfiguration(LLMConfigurationBase):
    """
    A configuration class for the LLMFallback class.
    """

    def __init__(self, *, llm_config: T_Configuration, llm_fallback_config: T_Configuration) -> None:
        super().__init__()
        self._llm_config = llm_config
        self._llm_fallback_config = llm_fallback_config

    def model_name(self) -> str:
        return f"{self._llm_config.model_name()} with fallback_{self._llm_fallback_config.model_name()}"


class LLMFallback(LLMBase[LLMFallbackConfiguration]):
    """
    A class that combines two language models, using a fallback mechanism upon failure.

    Attributes:
        _llm (LLMBase): The primary language model instance.
        _fallback (LLMBase): The fallback language model instance.
        _retry_before_fallback (int): The number of retry attempts with the primary language model
            before switching to the fallback.

    """

    def __init__(self, llm: LLMBase, fallback: LLMBase, retry_before_fallback: int = 2) -> None:
        config = LLMFallbackConfiguration(llm_config=llm.configuration, llm_fallback_config=fallback.configuration)
        super().__init__(configuration=config)

        self._llm = self.new_monitor("primary", llm)
        self._fallback = self.new_monitor("fallback", fallback)
        self._retry_before_fallback = retry_before_fallback

    @property
    def llm(self) -> LLMBase:
        return self._llm.inner

    @property
    def fallback(self) -> LLMBase:
        return self._fallback.inner

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        try:
            return self._llm_call_with_retry(context, messages, **kwargs)
        except Exception as base_exception:
            try:
                return self.fallback.post_chat_request(context.new_for(self._fallback), messages, **kwargs)
            except Exception as e:
                raise e from base_exception

    def _llm_call_with_retry(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        retry_count = 0
        while retry_count == 0 or retry_count < self._retry_before_fallback:
            try:
                return self.llm.post_chat_request(context, messages, **kwargs)
            except LLMCallException as e:
                retry_count += 1
                if self._is_retryable(e.code) and retry_count < self._retry_before_fallback:
                    time.sleep(1.25**retry_count)
                else:
                    raise e
            except Exception:
                raise
        raise LLMException(message=f"Main LLM failed after {retry_count} retries", llm_name=self._llm.name)

    @staticmethod
    def _is_retryable(code: int) -> bool:
        return code == 408 or code == 429 or code == 503 or code == 504
