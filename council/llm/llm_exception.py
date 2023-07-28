class LLMException(Exception):
    """
    Custom exception raised when the Large Language mModel is executed.
    """

    pass


class LLMTokenLimitException(Exception):
    """
    Custom exception raised when the number of tokens exceed the model limit.
    """

    def __init__(self, token_count: int, limit: int, model: str):
        """
        Initializes an instance of LLMTokenLimitException.

        Parameters:
            token_count (int): the actual number of tokens
            limit (int): the model limit
            model (str): the model

        Returns:
            None
        """
        super().__init__(f"token_count={token_count} is exceeding model {model} limit of {limit} tokens.")
