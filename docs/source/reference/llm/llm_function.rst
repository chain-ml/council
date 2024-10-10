LLMFunction
-----------

.. autoclass:: council.llm.LLMFunction

Here's how you can use LLMFunction for a sample SQL generation task.

.. testcode::

    from __future__ import annotations

    import os

    from council import OpenAILLM
    from council.llm import LLMParsingException, LLMResponse
    from council.llm.llm_function import LLMFunction
    from council.utils.code_parser import CodeParser

    SYSTEM_PROMPT = """
    You are a SQL expert producing SQL query to answer user question.

    # Instructions
    - Assess whether the question is reasonable and possible to solve
    given the database schema.
    - Follow `Response format` for output format
    - Always use LIMIT in your SQL query

    # Dataset info

    The dataset contains information about Airbnb listings.

    Table Name: listings

    ### Columns
    For each column, the name and data type are given as follows:
    {name}: {data type}
    name: TEXT
    price: INTEGER

    # Response format

    Your entire response must be inside the following code blocks.
    All code blocks are mandatory.

    ```solved
    True/False, indicating whether the task is solved based on the provided database schema
    ```

    ```sql
    SQL query answering the question if the task could be solved; leave empty otherwise
    ```
    """


    # Define a response type object with from_response() method
    class SQLResult:
        def __init__(self, solved: bool, sql: str) -> None:
            self.solved = solved
            self.sql = sql

        @staticmethod
        def from_response(response: LLMResponse) -> SQLResult:
            response_str = response.value
            solved_block = CodeParser.find_first("solved", response_str)
            if solved_block is None:
                raise LLMParsingException("No `solved` code block found!")

            solved = solved_block.code.lower() == "true"
            if not solved:
                return SQLResult(solved=False, sql="")

            sql_block = CodeParser.find_first("sql", response_str)
            if sql_block is None:
                raise LLMParsingException("No `sql` code block found!")

            sql = sql_block.code

            if "limit" not in sql.lower():
                raise LLMParsingException("Generated SQL query should contain a LIMIT clause")

            return SQLResult(solved=True, sql=sql)


    os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
    os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
    llm = OpenAILLM.from_env()

    # Create a function based on LLM, response parser and system prompt
    llm_function: LLMFunction[SQLResult] = LLMFunction(
        llm, SQLResult.from_response, SYSTEM_PROMPT
    )

    # Execute a function with user input
    response = llm_function.execute(
        user_message="Show me first 5 rows of the dataset ordered by price"
    )
    print(type(response))
    print(response.sql)

You can simplify this example with :class:`council.llm.llm_response_parser.CodeBlocksResponseParser`.
