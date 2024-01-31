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

    from council.llm import AzureLLM, LLMConfigObject
    
    llm_config = LLMConfigObject.from_yaml("filename.yaml")
    llm = AzureLLM.from_config(llm_config) 
```

```yaml
kind: LLMConfig
version: 0.1
metadata:
  name: an-openai-deployed-model
  labels:
    provider: OpenAI
spec:
  description: "Model used to do ABC"
  provider:
    name: CML-OpenAI
    openAISpec:
      model: gpt-4-1106-preview
      timeout: 60
      apiKey:
        fromEnvVar: OPENAI_API_KEY
  parameters:
    n: 3
    temperature: 0.5
```