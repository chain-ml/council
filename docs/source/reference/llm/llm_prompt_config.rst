LLMPromptConfigObject
---------------------

.. autoclass:: council.prompt.LLMPromptConfigObject

The following code illustrates the way to load prompt from a YAML file.

.. testcode::

    from council.prompt import LLMPromptConfigObject

    prompt = LLMPromptConfigObject.from_yaml("data/prompts/llm-prompt-sql-template.yaml")
    system_prompt = prompt.get_system_prompt_template("default")
    user_prompt = prompt.get_user_prompt_template("default")

.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template.yaml
    :language: yaml
