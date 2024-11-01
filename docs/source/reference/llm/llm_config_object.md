# LLMConfigObject

```{eval-rst}
.. autoclasstree:: council.llm.LLMConfigObject
    :full:
    :namespace: council

.. autoclass:: council.llm.LLMConfigObject
```

## Code Example

The following code illustrates the way an LLM could be loaded from a YAML file with a specific provider.

```{eval-rst}
.. testcode::

    from council.llm import OpenAILLM, LLMConfigObject
    
    llm_config = LLMConfigObject.from_yaml("data/configs/llm-config-openai.yaml")
    llm = OpenAILLM.from_config(llm_config)
```

Or use `council.llm.get_llm_from_config` to determine provider class automatically based on config file.

```{eval-rst}
.. testcode::

    from council.llm import get_llm_from_config

    llm = get_llm_from_config("data/configs/llm-config-openai.yaml")
```

## OpenAI Config Example

```{eval-rst}
.. literalinclude:: ../../../data/configs/llm-config-openai.yaml
    :language: yaml
```

## Anthropic Config Example

```{eval-rst}
.. literalinclude:: ../../../data/configs/llm-config-anthropic.yaml
    :language: yaml
```

## Gemini Config Example

```{eval-rst}
.. literalinclude:: ../../../data/configs/llm-config-gemini.yaml
    :language: yaml
```

## Azure Config Example

```{eval-rst}
.. literalinclude:: ../../../data/configs/llm-config-azure.yaml
    :language: yaml
```

## Fallback Config Example

**Note** that `provider` and `fallbackProvider` can be any providers from above.

```{eval-rst}
.. literalinclude:: ../../../data/configs/llm-config-with-fallback.yaml
    :language: yaml
```
