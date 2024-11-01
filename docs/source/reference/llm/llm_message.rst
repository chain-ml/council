LLMMessage
----------

.. autoclass:: council.llm.LLMMessage
   :member-order: bysource

LLMMessageData
--------------

.. autoclass:: council.llm.llm_message.LLMMessageData

LLMCacheControlData
-------------------

.. autoclass:: council.llm.llm_message.LLMCacheControlData
    :no-inherited-members:

Code Example
^^^^^^^^^^^^

Here's how you can use Anthropic prompt caching with council.

.. code-block:: python

    import os

    # !pip install council-ai==0.0.24

    from council.llm import AnthropicLLM
    from council.llm.llm_message import LLMMessage, LLMCacheControlData
    from council.contexts import LLMContext

    os.environ["ANTHROPIC_API_KEY"] = "sk-YOUR-KEY-HERE"
    os.environ["ANTHROPIC_LLM_MODEL"] = "claude-3-haiku-20240307"

    # Ensure that the number of tokens in a cacheable message exceeding
    # the minimum cacheable token count, which is 2048 for Haiku;
    # otherwise, the message will not be cached.
    HUGE_STATIC_SYSTEM_PROMPT = ""

    # Create a system message with ephemeral caching
    system_message_with_cache = LLMMessage.system_message(
        HUGE_STATIC_SYSTEM_PROMPT,
        data=[LLMCacheControlData.ephemeral()]
    )

    # Initialize the messages list with cachable system message
    messages = [
        system_message_with_cache,
        LLMMessage.user_message("What are benefits of using caching?")
    ]

    llm = AnthropicLLM.from_env()

    result = llm.post_chat_request(LLMContext.empty(), messages)
    print(result.first_choice)
    print(result.raw_response["usage"])
