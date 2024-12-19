# EchoResponseParser

```{eval-rst}
.. autoclass:: council.llm.EchoResponseParser
```

# StringResponseParser

```{eval-rst}
.. autoclass:: council.llm.StringResponseParser
```

# BaseModelResponseParser

```{eval-rst}
.. autoclass:: council.llm.BaseModelResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

# CodeBlocksResponseParser

```{eval-rst}
.. autoclass:: council.llm.CodeBlocksResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

## Code Example

Here's how you can simplify {class}`council.llm.LLMFunction` example for a sample SQL generation task.

```python
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
```

# YAMLBlockResponseParser

```{eval-rst}
.. autoclass:: council.llm.YAMLBlockResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

## Code Example

````python
import os
from typing import Literal

# !pip install council-ai==0.0.27

from council import OpenAILLM
from council.llm.llm_function import LLMFunction
from council.llm.llm_response_parser import YAMLBlockResponseParser
from pydantic import Field

SYSTEM_PROMPT = """
Generate RPG character:

{response_template}
"""


class RPGCharacterFromYAMLBlock(YAMLBlockResponseParser):
    character_class: Literal["Warrior", "Mage", "Rogue", "Bard", "Tech Support"] = Field(
        ..., description="Character's class (Warrior, Mage, Rogue, Bard or Tech Support)"
    )
    name: str = Field(..., min_length=3, description="Character's name")
    description: str = Field(..., min_length=50, description="Character's tragic backstory, 50 chars minimum")
    health: int = Field(..., ge=1, le=100, description="Character's health, integer, from 1 to 100 points")


os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
llm = OpenAILLM.from_env()

llm_function: LLMFunction[RPGCharacterFromYAMLBlock] = LLMFunction(
    llm,
    RPGCharacterFromYAMLBlock.from_response,
    SYSTEM_PROMPT.format(response_template=RPGCharacterFromYAMLBlock.to_response_template()),
)

character = llm_function.execute(user_message="Create some wise mage")
print(type(character))
print(f"{character.name}, {character.character_class} ({character.health}/100 hp)")
print(character.description)
````

# YAMLResponseParser

```{eval-rst}
.. autoclass:: council.llm.YAMLResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

# JSONBlockResponseParser

```{eval-rst}
.. autoclass:: council.llm.JSONBlockResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

# JSONResponseParser

```{eval-rst}
.. autoclass:: council.llm.JSONResponseParser
    :no-inherited-members:
    :exclude-members: model_computed_fields, model_config, model_fields
```

## Code Example

Usage example with OpenAI json mode:

```python
import os
from typing import Literal

# !pip install council-ai==0.0.27

from council import OpenAILLM
from council.llm.llm_function import LLMFunction
from council.llm.llm_response_parser import JSONResponseParser
from pydantic import Field

SYSTEM_PROMPT = """
Generate RPG character:

{response_template}
"""


class RPGCharacterFromJSON(JSONResponseParser):
    character_class: Literal["Warrior", "Mage", "Rogue", "Bard", "Tech Support"] = Field(
        ..., description="Character's class (Warrior, Mage, Rogue, Bard or Tech Support)"
    )
    name: str = Field(..., min_length=3, description="Character's name")
    description: str = Field(..., min_length=50, description="Character's tragic backstory, 50 chars minimum")
    health: int = Field(..., ge=1, le=100, description="Character's health, integer, from 1 to 100 points")


os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY-HERE"
os.environ["OPENAI_LLM_MODEL"] = "gpt-4o-mini-2024-07-18"
llm = OpenAILLM.from_env()

llm_function: LLMFunction[RPGCharacterFromJSON] = LLMFunction(
    llm,
    RPGCharacterFromJSON.from_response,
    SYSTEM_PROMPT.format(response_template=RPGCharacterFromJSON.to_response_template()),
)

character = llm_function.execute(
    user_message="Create some strong warrior",
    response_format={"type": "json_object"}  # using OpenAI's json mode
)
print(type(character))
print(f"{character.name}, {character.character_class} ({character.health}/100 hp)")
print(character.description)
```
