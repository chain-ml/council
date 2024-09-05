class LLModels:
    Azure: str = "azure-llmodel.yaml"
    OpenAI: str = "openai-llmodel.yaml"
    Anthropic: str = "anthropic-llmodel.yaml"
    AzureWithFallback: str = "azure-with-fallback-llmodel.yaml"
    Gemini: str = "gemini-llmodel.yaml"


class LLMPrompts:
    sample: str = "prompt-abc.yaml"
    sql: str = "prompt-sql.yaml"
    sql_template: str = "prompt-template-sql.yaml"
