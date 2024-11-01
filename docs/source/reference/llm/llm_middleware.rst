LLMMiddleware
-------------

.. autoclass:: council.llm.LLMMiddleware

LLMMiddlewareChain
------------------

.. autoclass:: council.llm.LLMMiddlewareChain

LLMLoggingMiddleware
--------------------

.. autoclass:: council.llm.LLMLoggingMiddleware

LLMFileLoggingMiddleware
------------------------

.. autoclass:: council.llm.LLMFileLoggingMiddleware

LLMRetryMiddleware
------------------

.. autoclass:: council.llm.LLMRetryMiddleware

LLMCachingMiddleware
--------------------

.. autoclass:: council.llm.LLMCachingMiddleware

Code Example
^^^^^^^^^^^^

Example usage with :class:`council.llm.LLMFunction`.

.. code-block:: python

    import dotenv

    # !pip install council-ai==0.0.26

    from council import AnthropicLLM
    from council.llm import LLMFunction, LLMCachingMiddleware, LLMResponse
    from council.llm.llm_response_parser import EchoResponseParser


    dotenv.load_dotenv()
    llm = AnthropicLLM.from_env()
    llm_func: LLMFunction[LLMResponse] = LLMFunction(
        llm,
        EchoResponseParser.from_response,
        system_message="You're a helpful assistant"
    )
    # add caching middleware
    llm_func.add_middleware(LLMCachingMiddleware())

    # first request will be cached
    llm_response_v1 = llm_func.execute("What is the capital of France?")
    print(llm_response_v1.duration)  # 0.43
    for consumption in llm_response_v1.result.consumptions:
        print(consumption)
    # sample output:
    # claude-3-haiku-20240307 consumption: 1 call
    # claude-3-haiku-20240307 consumption: 0.3583 second
    # claude-3-haiku-20240307:prompt_tokens consumption: 19 token
    # ...
    # claude-3-haiku-20240307:total_tokens_cost consumption: 1.852e-05 USD

    # will hit the cache
    llm_response_v1_1 = llm_func.execute("What is the capital of France?")
    print(llm_response_v1_1.duration)  # 0
    for consumption in llm_response_v1_1.result.consumptions:
        print(consumption)
    # sample output:
    # claude-3-haiku-20240307 consumption: 1 cached_call
    # claude-3-haiku-20240307 consumption: 0.3583 cached_second
    # claude-3-haiku-20240307:prompt_tokens consumption: 19 cached_token
    # ...
    # claude-3-haiku-20240307:total_tokens_cost consumption: 1.852e-05 cached_USD

    # will not hit the cache since message is different
    llm_response_v2 = llm_func.execute("Again, what is the capital of France?")


LLMRequest
-----------

.. autoclass:: council.llm.LLMRequest

LLMResponse
-----------

.. autoclass:: council.llm.LLMResponse
