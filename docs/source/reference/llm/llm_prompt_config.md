# LLMPromptConfigObject

```{eval-rst}
.. autoclass:: council.prompt.LLMPromptConfigObject
```

## Code Example

The following code illustrates the way to load prompt from a YAML file.
```{eval-rst}
.. testcode::

    from council.prompt import LLMPromptConfigObject

    prompt = LLMPromptConfigObject.from_yaml("data/prompts/llm-prompt-sql-template.yaml")
    system_prompt = prompt.get_system_prompt_template("default")
    user_prompt = prompt.get_user_prompt_template("default")
```

Sample yaml file:

```{eval-rst}
.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template.yaml
    :language: yaml
```
