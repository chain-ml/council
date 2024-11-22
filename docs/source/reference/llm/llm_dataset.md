# LLMDatasetMessage

```{eval-rst}
.. autoclass:: council.prompt.LLMDatasetMessage
```

# LLMDatasetConversation

```{eval-rst}
.. autoclass:: council.prompt.LLMDatasetConversation
```

# LLMDataset

```{eval-rst}
.. autoclass:: council.prompt.LLMDatasetObject
   :member-order: bysource
```

## Fine-tuning

Here's an example of LLMDataset YAML file for fine-tuning:

```{eval-rst}
.. literalinclude:: ../../../data/datasets/llm-dataset-fine-tuning.yaml
    :language: yaml
```

With this code, you can load a dataset from a YAML file and save it as a JSONL file for [OpenAI fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning):

```{eval-rst}
.. testcode::
    
    import os
    import tempfile
    from council.prompt import LLMDatasetObject
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "dataset.jsonl")

        dataset = LLMDatasetObject.from_yaml("data/datasets/llm-dataset-fine-tuning.yaml")
        dataset.save_jsonl_messages(tmp_file)
    
        lines = LLMDatasetObject.read_jsonl(tmp_file)
        for line in lines:
            print(line)
```

This will produce the following JSONL file:

```{eval-rst}
.. testoutput::

    {'messages': [{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}, {'role': 'user', 'content': 'I fell off my bike today.'}, {'role': 'assistant', 'content': "It's great that you're getting exercise outdoors!"}]}
    {'messages': [{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}, {'role': 'user', 'content': 'I lost my tennis match today.'}, {'role': 'assistant', 'content': "It's ok, it happens to everyone."}, {'role': 'user', 'content': 'But I trained so hard!'}, {'role': 'assistant', 'content': 'It will pay off next time.'}, {'role': 'user', 'content': "I'm going to switch to golf."}, {'role': 'assistant', 'content': 'Golf is fun too!'}, {'role': 'user', 'content': "I don't even know how to play golf."}, {'role': 'assistant', 'content': "It's easy to learn!"}]}
    {'messages': [{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}, {'role': 'user', 'content': 'I lost my book today.'}, {'role': 'assistant', 'content': 'You can read everything on ebooks these days!'}]}
    {'messages': [{'role': 'system', 'content': 'You are a happy assistant that puts a positive spin on everything.'}, {'role': 'user', 'content': "I'm hungry."}, {'role': 'assistant', 'content': 'Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!Eat a banana!'}]}
```

## Few-shot examples

You can use the same dataset to manage few-shot examples and format them by calling `dataset.format_examples()`.

```{eval-rst}
.. testcode::
    
    import os
    import tempfile
    from council.prompt import LLMDatasetObject
    
    dataset = LLMDatasetObject.from_yaml("data/datasets/llm-dataset-fine-tuning.yaml")
    examples = dataset.format_examples(
        start_prefix="### Example {i} ###", 
        end_prefix="### End Example {i} ###"
    )
    
    print(examples[0])
```

```{eval-rst}
.. testoutput::

    ### Example 1 ###
    user: I fell off my bike today.
    assistant: It's great that you're getting exercise outdoors!
    ### End Example 1 ###
```

## Batch API

Here's an example of LLMDataset YAML file for batch API:

```{eval-rst}
.. literalinclude:: ../../../data/datasets/llm-dataset-batch.yaml
    :language: yaml
```

With this code, you can load a dataset from a YAML file and save it as a JSONL file for [OpenAI batch API](https://platform.openai.com/docs/guides/batch):

```{eval-rst}
.. testcode::

    import os
    import tempfile
    from council.prompt import LLMDatasetObject
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "batch.jsonl")

        dataset = LLMDatasetObject.from_yaml("data/datasets/llm-dataset-batch.yaml")
        dataset.save_jsonl_requests(tmp_file, model="gpt-4o-mini")
    
        lines = LLMDatasetObject.read_jsonl(tmp_file)
        for line in lines:
            print(line)
```

This will produce the following JSONL file:

```{eval-rst}
.. testoutput::

    {'custom_id': 'request-0', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': 'I had a wonderful day at the park with my family.'}]}}
    {'custom_id': 'request-1', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': 'The weather was okay, not too bad, not too great.'}]}}
    {'custom_id': 'request-2', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': 'My car broke down on the way to work, and it ruined my entire day.'}]}}
    {'custom_id': 'request-3', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': "I received a promotion at work today, and I'm feeling ecstatic!"}]}}
    {'custom_id': 'request-4', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': "The movie was average; it wasn't what I expected."}]}}
    {'custom_id': 'request-5', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': [{'role': 'system', 'content': 'Classify the sentiment of user inputs into one of three categories: \npositive, neutral, or negative. \nRespond with just the sentiment label.'}, {'role': 'user', 'content': 'I missed my flight and had to reschedule everything, which was frustrating.'}]}}
```

# LLMDatasetValidator

```{eval-rst}
.. autoclass:: council.prompt.LLMDatasetValidator
```
