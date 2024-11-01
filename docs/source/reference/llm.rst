LLM
===

.. autoclasstree:: council.llm
    :zoom:
    :full:
    :namespace: council

.. automodule:: council.llm

Overview
--------

The `council.llm` module provides a unified interface for interacting with various LLM providers, along with tools for handling responses, caching, logging and tracking consumptions easily.

LLMs
~~~~

Create your LLM easily from yaml :class:`~council.llm.LLMConfigObject` (see for different config examples).

Supported providers are: :class:`~council.llm.OpenAILLM`, :class:`~council.llm.AnthropicLLM`, :class:`~council.llm.GeminiLLM`, :class:`~council.llm.AzureLLM`.

.. testcode::

    from council.llm import get_llm_from_config

    # will adjust provider class automatically based on config file
    llm = get_llm_from_config("data/configs/llm-config-openai.yaml")

Post Chat Request and Cost Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use `llm.post_chat_request` to send a request to LLM. The object you will get back (:class:`~council.llm.LLMResult`) contains LLM response as well as list of :class:`~council.contexts.Consumption` associated with the call, including duration, token usage and cost.

.. code-block:: python

    import dotenv
    from council import LLMContext
    from council.llm import LLMMessage, get_llm_from_config

    llm = get_llm_from_config("data/configs/llm-config-openai.yaml")
    result = llm.post_chat_request(
        LLMContext.empty(),
        messages=[LLMMessage.user_message("Hello world")]
    )

    print(result.first_choice)
    # sample output:
    # Hello! How can I assist you today?

    for consumption in result.consumptions:
        print(consumption)
    # sample output:
    # gpt-4o-mini-2024-07-18 consumption: 1 call
    # gpt-4o-mini-2024-07-18 consumption: 0.9346139430999756 second
    # gpt-4o-mini-2024-07-18:prompt_tokens consumption: 9 token
    # gpt-4o-mini-2024-07-18:completion_tokens consumption: 9 token
    # gpt-4o-mini-2024-07-18:total_tokens consumption: 18 token
    # gpt-4o-mini-2024-07-18:prompt_tokens_cost consumption: 1.3499e-06 USD
    # gpt-4o-mini-2024-07-18:completion_tokens_cost consumption: 5.399e-06 USD
    # gpt-4o-mini-2024-07-18:total_tokens_cost consumption: 6.7499e-06 USD

Anthropic Prompt Caching
^^^^^^^^^^^^^^^^^^^^^^^^

See :class:`~council.llm.llm_message.LLMCacheControlData` to see an example of how to enable Anthropic prompt caching.

LLM Functions
~~~~~~~~~~~~~

LLM Functions provide structured ways to interact with LLMs including built-in response parsing, error handling and retries.

- See :class:`~council.llm.LLMFunction`
- Or create :class:`~council.llm.LLMFunctionWithPrompt` with :class:`~council.prompt.LLMPromptConfigObject`

Response Parsers
~~~~~~~~~~~~~~~~

To use LLMFunctions conveniently, you can leverage response parsers to parse common response formats automatically.

- Raw response: :class:`~council.llm.llm_response_parser.EchoResponseParser`
- Plain test: :class:`~council.llm.llm_response_parser.StringResponseParser`
- Code blocks: :class:`~council.llm.llm_response_parser.CodeBlocksResponseParser`
- YAML responses in code block or plain text: :class:`~council.llm.llm_response_parser.YAMLBlockResponseParser` and :class:`~council.llm.llm_response_parser.YAMLResponseParser`
- JSON responses in code block or plain text: :class:`~council.llm.llm_response_parser.JSONBlockResponseParser` and :class:`~council.llm.llm_response_parser.JSONResponseParser`

LLMMiddleware
~~~~~~~~~~~~~

Enhance LLM interactions with middleware components that could modify requests and responses introducing custom logic, such as logging, caching, updating LLM configs, etc.

Core middlewares:

- :class:`~council.llm.LLMCachingMiddleware`
- :class:`~council.llm.LLMLoggingMiddleware` and :class:`~council.llm.LLMFileLoggingMiddleware`

Middleware management:

- :class:`~council.llm.LLMMiddlewareChain`
- :class:`~council.llm.LLMMiddleware`

Reference
---------

.. toctree::
    :maxdepth: 1
    :glob:

    llm/*
