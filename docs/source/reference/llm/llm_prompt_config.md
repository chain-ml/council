# PromptTemplateBase

```{eval-rst}
.. autoclass:: council.prompt.PromptTemplateBase
```

# StringPromptTemplate

```{eval-rst}
.. autoclass:: council.prompt.StringPromptTemplate
```

## Example

```{eval-rst}
.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template.yaml
    :language: yaml
```

# XMLPromptTemplate

```{eval-rst}
.. autoclass:: council.prompt.XMLPromptTemplate
```

## XMLPromptSection

```{eval-rst}
.. autoclass:: council.prompt.XMLPromptSection
```

## Example

```{eval-rst}
.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template-xml.yaml
    :language: yaml
```

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
