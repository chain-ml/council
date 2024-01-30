import os


class LLModels:
    Azure: str = "azure-llmodel.yaml"
    OpenAI: str = "openai-llmodel.yaml"
    Anthropic: str = "anthropic-llmodel.yaml"
    AzureWithFallback: str = "azure-with-fallback-llmodel.yaml"


def get_data_filename(filename: str):
    return os.path.join(os.path.dirname(__file__), "..", "data", filename)
