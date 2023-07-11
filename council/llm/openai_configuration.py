from council.utils import read_env


class OpenAIConfiguration:
    """
    Configuration for :class:OpenAILLM

    Args:
        api_key (str): the OpenAI api key
        model: optional model version to use
        temperature (str): optional temperature settings for the LLM
        max_tokens (str): optional limit on number of tokens
        top_p (str): optional he model only takes into account the tokens with the highest probability mass
        n (str): optional How many completions to generate for each prompt
        presence_penalty (str): optional, impacts how the model penalizes new tokens based on whether
            they have appeared in the text so far
        frequency_penalty (str): optional, impacts how the model penalizes new tokens based on their existing
            frequency in the text.
        timeout: int - seconds to wait for response from OpenAI before timing out

    Notes:
        * see https://platform.openai.com/docs/api-reference/chat
    """

    api_key: str
    authorization: str  # not a parameter - used to optimize calls
    model: str
    max_tokens: str
    temperature: str
    top_p: str
    n: str
    presence_penalty: str
    frequency_penalty: str
    timeout: int = 30

    @classmethod
    def from_env(cls):
        config = OpenAIConfiguration()
        config.api_key = read_env("OPENAI_API_KEY")
        config.authorization = f"Bearer {config.api_key}"
        config.model = read_env("OPENAI_LLM_MODEL", required=False)

        config.temperature = read_env("OPENAI_LLM_TEMPERATURE", required=False)
        config.top_p = read_env("OPENAI_LLM_TOP_P", required=False)
        config.n = read_env("OPENAI_LLM_N", required=False)

        config.presence_penalty = read_env("OPENAI_LLM_PRESENCE_PENALTY", required=False)
        config.frequency_penalty = read_env("OPENAI_LLM_FREQUENCY_PENALTY", required=False)

        timeout = read_env("OPENAI_LLM_TIMEOUT", required=False, default="30")
        config.timeout = int(timeout)

        return config
