"""

A module that provides a class `LLMFallback` which serves as a fallback wrapper for two `LLMBase` instances.

The `LLMFallback` class is responsible for implementing a mechanism that first attempts to use a primary
LLM (Language Learning Model) instance for processing requests. If the primary LLM fails to handle the request,
the class then tries to use a fallback LLM. Additionally, it includes logic to retry the request with the
primary LLM a specified number of times before switching to the fallback LLM. The class also contains methods for
identifying retryable error codes and performing chat requests with retries.

Attributes:
    _llm (Monitored[LLMBase]): A monitored wrapper for the primary LLM instance.
    _fallback (Monitored[LLMBase]): A monitored wrapper for the fallback LLM instance.
    _retry_before_fallback (int): The number of times to retry the primary LLM before using the fallback instance.

Methods:
    __init__(self, llm: LLMBase, fallback: LLMBase, retry_before_fallback: int=2) -> None:
        Initializes the `LLMFallback` instance with a primary LLM, a fallback LLM, and the number of retries
        before using the fallback.

    llm(self) -> LLMBase:
        Returns the primary LLM instance.

    fallback(self) -> LLMBase:
        Returns the fallback LLM instance.

    _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        Attempts to post a chat request using the primary LLM. If it fails, attempts the request with the fallback
        LLM, propagating exceptions from the primary LLM as needed.

    _llm_call_with_retry(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        Attempts to post a chat request with the primary LLM and retries a specified number of times before failing.

    _is_retryable(code: int) -> bool:
        A static method that determines if an error code is retryable. It identifies network-related errors such
        as timeouts and rate limits that are appropriate for a retry.


"""
import time
from typing import Any, Sequence

from council.contexts import LLMContext, Monitored
from council.llm import LLMBase, LLMCallException, LLMException, LLMMessage, LLMResult


class LLMFallback(LLMBase):
    """
    Provides a wrapper class for a Language Model (LLM) with a fallback mechanism.
    This class takes two language model instances: a primary LLM and a fallback LLM. It attempts to process
    requests using the primary LLM, and if that fails after a specified number of retries, it then attempts
    to process the same request using the fallback LLM.
    
    Attributes:
        _llm (Monitored[LLMBase]):
             A monitored instance of the primary language model.
        _fallback (Monitored[LLMBase]):
             A monitored instance of the fallback language model.
        _retry_before_fallback (int):
             The number of retry attempts for the primary LLM before switching to the fallback.
    
    Methods:
        __init__:
             Constructor for initializing the LLMFallback class with primary and fallback LLMs.
        llm:
             Property that returns the instance of the primary LLM.
        fallback:
             Property that returns the instance of the fallback LLM.
        _post_chat_request:
             Handles the logic of posting a chat request, with retries on the primary LLM and
            fallback to the secondary LLM upon failure.
        _llm_call_with_retry:
             Tries to make a call to the primary LLM with a certain number of retries.
        _is_retryable:
             A static method determining if an error code is considered retryable.
    
    Raises:
        LLMCallException:
             If all retries on the primary LLM fail.
        LLMException:
             If the main LLM fails after the specified number of retries.
        

    """

    _llm: Monitored[LLMBase]
    _fallback: Monitored[LLMBase]

    def __init__(self, llm: LLMBase, fallback: LLMBase, retry_before_fallback: int = 2) -> None:
        """
        Initializes the instance with a primary LLMBase and fallback LLMBase, along with a specified
        number of retries to attempt before falling back.
        This constructor sets up monitors for both the primary and fallback language learning models
        (LLMBase) and initializes the number of retries before using the fallback model. This is
        done by wrapping the provided LLMBase instances with a monitoring utility through the
        `new_monitor` method and storing them along with the retry count.
        
        Parameters:
            llm (LLMBase):
                 The primary language learning model to be used.
            fallback (LLMBase):
                 The fallback language learning model to be used if the primary model fails.
            retry_before_fallback (int, optional):
                 The number of times to retry the primary model
                before resorting to the fallback model. Defaults to 2.
            

        """
        super().__init__()
        self._llm = self.new_monitor("primary", llm)
        self._fallback = self.new_monitor("fallback", fallback)
        self._retry_before_fallback = retry_before_fallback

    @property
    def llm(self) -> LLMBase:
        """
        Retrieves the inner LLMBase instance associated with this object.
        
        Returns:
            (LLMBase):
                 The inner LLMBase instance.
        
        Raises:
            AttributeError:
                 If the inner LLMBase instance does not exist or is inaccessible.

        """
        return self._llm.inner

    @property
    def fallback(self) -> LLMBase:
        """
        Property that gets the inner fallback LLMBase instance.
        This property provides access to the inner object of type LLMBase that serves as a fallback,
        allowing for a graceful degradation or alternative behavior when the primary functionality
        is not available or appropriate.
        
        Returns:
            (LLMBase):
                 The LLMBase instance that is designated as the fallback.
            

        """
        return self._fallback.inner

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the language model and handles exceptions by
        falling back to an alternative service if the primary call fails.
        This method sends a request to the language model. If an exception occurs,
        it will try to resend the request to a fallback mechanism. If the fallback
        also raises an exception, it will be raised with the original exception as
        its context.
        
        Args:
            context (LLMContext):
                 The context for the language model conversation.
            messages (Sequence[LLMMessage]):
                 A sequence of message objects to be sent
                to the language model.
            **kwargs (Any):
                 Additional keyword arguments that may be necessary for
                the underlying `_llm_call_with_retry` method or the fallback post
                chat request.
        
        Returns:
            (LLMResult):
                 The result of the chat request, which may come from either
                the primary call or the fallback service.
        
        Raises:
            Exception:
                 Reraised from the base exception if both the primary call
                and the fallback fail, preserving the exception chain.

        """
        try:
            return self._llm_call_with_retry(context, messages, **kwargs)
        except Exception as base_exception:
            try:
                return self.fallback.post_chat_request(context.new_for(self._fallback), messages, **kwargs)
            except Exception as e:
                raise e from base_exception

    def _llm_call_with_retry(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Attempts to call the LLM method post_chat_request with retries on failure.
        This method attempts to send a request to an LLM (Language Learning Model) by repeatedly calling
        the `post_chat_request` method with the provided context and messages. If a call fails
        with an LLMCallException and the error code is determined to be retryable, it will
        wait for an exponential backoff period and retry the call up to a maximum number
        specified by the `_retry_before_fallback` attribute. If the maximum number of retries
        is reached or the exception is not retryable, the method will raise either LLMCallException
        or LLMException, respectively.
        
        Args:
            context (LLMContext):
                 The context in which the LLM request is being made.
            messages (Sequence[LLMMessage]):
                 A sequence of messages to be sent to the LLM.
            **kwargs (Any):
                 Additional keyword arguments to be passed to the `post_chat_request` method.
        
        Returns:
            (LLMResult):
                 The result returned by the LLM upon success.
        
        Raises:
            LLMCallException:
                 If a retryable error occurs but retries are exhausted or a non-retryable
                error occurs.
            LLMException:
                 If the LLM fails to process the request after the specified number of retries.

        """
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
        """
        Determines if an HTTP response status code indicates a retryable error condition.
        This method evaluates the given HTTP response status code to ascertain if it
        represents a condition where the request can be retried. Typical retryable
        status codes include 408 Request Timeout, 429 Too Many Requests, 503 Service
        Unavailable, and 504 Gateway Timeout. This is usually the case when the server
        is temporarily unable to handle the request due to a temporary overloading or
        maintenance.
        
        Args:
            code (int):
                 The HTTP status code to evaluate for retryability.
        
        Returns:
            (bool):
                 True if the status code is one of the retryable errors, False otherwise.

        """
        return code == 408 or code == 429 or code == 503 or code == 504
