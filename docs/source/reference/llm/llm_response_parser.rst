BaseModelResponseParser
-----------------------

.. autoclass:: council.llm.llm_response_parser.BaseModelResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

CodeBlocksResponseParser
------------------------

.. autoclass:: council.llm.llm_response_parser.CodeBlocksResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

Here's how you can simplify :class:`council.llm.LLMFunction` example for a sample SQL generation task.

.. code-block:: python

    import os

    # !pip install council-ai==0.0.24

    from council import OpenAILLM
    from council.llm import LLMParsingException
    from council.llm.llm_function import LLMFunction
    from council.llm.llm_response_parser import CodeBlocksResponseParser

    SYSTEM_PROMPT = "same system prompt as in LLMFunction example"


    # CodeBlocksResponseParser will provide from_response() automatically for you
    class SQLResultFromCodeBlocks(CodeBlocksResponseParser):
        solved: bool
        sql: str

        def validator(self) -> None:
            if "limit" not in self.sql.lower():
                raise LLMParsingException("Generated SQL query should contain a LIMIT clause")


    os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
    os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
    llm = OpenAILLM.from_env()

    # All the remaining code stays the same
    llm_function: LLMFunction[SQLResultFromCodeBlocks] = LLMFunction(
        llm, SQLResultFromCodeBlocks.from_response, SYSTEM_PROMPT
    )

    response = llm_function.execute(
        user_message="Show me first 5 rows of the dataset ordered by price"
    )
    print(type(response))
    print(response.sql)

YAMLBlockResponseParser
-----------------------

.. autoclass:: council.llm.llm_response_parser.YAMLBlockResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

Usage example:

.. code-block:: python

    import os
    from typing import Literal

    # !pip install council-ai==0.0.24

    from council import OpenAILLM
    from council.llm.llm_function import LLMFunction
    from council.llm.llm_response_parser import YAMLBlockResponseParser
    from pydantic import Field

    SYSTEM_PROMPT = """
    Output RPG character info in the following YAML block:

    ```yaml
    character_class: # character's class (Warrior, Mage, Rogue, Bard or Tech Support)
    name: # character's name
    description: # character's tragic backstory, 50 chars minimum
    health: # character's health, integer, from 1 to 100 points
    ```
    """


    class RPGCharacterFromYAMLBlock(YAMLBlockResponseParser):
        name: str
        character_class: Literal["Warrior", "Mage", "Rogue", "Bard", "Tech Support"]
        description: str = Field(..., min_length=50)
        health: int = Field(..., ge=1, le=100)


    os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
    os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
    llm = OpenAILLM.from_env()

    llm_function: LLMFunction[RPGCharacterFromYAMLBlock] = LLMFunction(
        llm, RPGCharacterFromYAMLBlock.from_response, SYSTEM_PROMPT
    )

    character = llm_function.execute(user_message="Create some wise mage")
    print(type(character))
    print(f"{character.name}, {character.character_class} ({character.health}/100 hp)")
    print(character.description)


YAMLResponseParser
------------------

.. autoclass:: council.llm.llm_response_parser.YAMLResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

JSONBlockResponseParser
-----------------------

.. autoclass:: council.llm.llm_response_parser.JSONBlockResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

JSONResponseParser
------------------

.. autoclass:: council.llm.llm_response_parser.JSONResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields

Usage example with OpenAI json mode:

.. code-block:: python

    import os
    from typing import Literal

    # !pip install council-ai==0.0.24

    from council import OpenAILLM
    from council.llm.llm_function import LLMFunction
    from council.llm.llm_response_parser import JSONResponseParser
    from pydantic import Field

    SYSTEM_PROMPT = """
    Output RPG character info in the following JSON format:

    {
    character_class: # character's class (Warrior, Mage, Rogue, Bard or Tech Support)
    name: # character's name
    description: # character's tragic backstory, 50 chars minimum
    health: # character's health, integer, from 1 to 100 points
    }
    """


    class RPGCharacterFromJSON(JSONResponseParser):
        name: str
        character_class: Literal["Warrior", "Mage", "Rogue", "Bard", "Tech Support"]
        description: str = Field(..., min_length=50)
        health: int = Field(..., ge=1, le=100)


    os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
    os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
    llm = OpenAILLM.from_env()

    llm_function: LLMFunction[RPGCharacterFromJSON] = LLMFunction(
        llm, RPGCharacterFromJSON.from_response, SYSTEM_PROMPT
    )

    character = llm_function.execute(
        user_message="Create some wise mage",
        response_format={"type": "json_object"}  # using OpenAI's json mode
    )
    print(type(character))
    print(f"{character.name}, {character.character_class} ({character.health}/100 hp)")
    print(character.description)
