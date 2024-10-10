LLMMessage
----------

.. autoclass:: council.llm.LLMMessage

LLMMessageData
--------------

.. autoclass:: council.llm.llm_message.LLMMessageData

LLMCacheControlData
-------------------

.. autoclass:: council.llm.llm_message.LLMCacheControlData
    :no-inherited-members:

Here's how you can use Anthropic prompt caching with council.

.. testcode::

    import os

    from council.llm import AnthropicLLM
    from council.llm.llm_message import LLMMessage, LLMCacheControlData
    from council.contexts import LLMContext

    os.environ["ANTHROPIC_API_KEY"] = "YOUR-KEY-HERE"
    os.environ["ANTHROPIC_LLM_MODEL"] = "claude-3-haiku-20240307"

    # Create a system message with ephemeral caching
    system_message_with_cache = LLMMessage.system_message(
        HUGE_STATIC_SYSTEM_PROMPT,
        data=[LLMCacheControlData.ephemeral()]
    )
    # Ensure that the number of tokens in a cacheable message exceeds
    # the minimum cacheable token count, which is 2048 for Haiku;
    # otherwise, the message will not be cached.

    # Initialize the messages list with cachable system message
    messages = [
        system_message_with_cache,
        LLMMessage.user_message("What are benefits of using caching?")
    ]

    llm = AnthropicLLM.from_env()

    result = llm.post_chat_request(LLMContext.empty(), messages)
    print(result.first_choice)
    print(result.raw_response["usage"])
