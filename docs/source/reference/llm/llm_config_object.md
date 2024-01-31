# LLMConfigObject

```{eval-rst}
.. autoclasstree:: council.llm.LLMConfigObject
    :full:
    :namespace: council
```

```{eval-rst}
.. autoclass:: council.llm.LLMConfigObject
```

The following code illustrates the way an LLM could be loaded from a YAML file.

```{eval-rst}
.. testcode::

    from council.llm import OpenAILLM, LLMConfigObject
    
    llm_config = LLMConfigObject.from_yaml("data/openai-llm-model.yaml")
    llm = OpenAILLM.from_config(llm_config) 
```

```{literalinclude} ../../../data/openai-llm-model.yaml
:language: yaml
```