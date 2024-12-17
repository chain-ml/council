from __future__ import annotations

import abc
import os.path
from typing import Protocol, TypeVar, Generic, Union, List, Tuple, Type, Sequence, Optional

import dotenv
from pydantic import Field, BaseModel
from typing_extensions import Self

from council import LLMContext, OpenAILLM
from council.llm import LLMResponse, LLMBase, LLMMessage, LLMMiddlewareChain, LLMRequest, LLMParsingException, \
    get_llm_from_config
from council.llm.llm_response_parser import LLMResponseParser, YAMLBlockResponseParser
from council.mocks import MockLLM



# Protocol is better here?
# FunctionInput and FunctionOutput as more general types
class LLMFunctionInput(abc.ABC):
    @abc.abstractmethod
    def to_prompt(self) -> str:
        """Convert the object to a prompt string."""
        ...


class LLMFunctionOutput(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_response(cls, response: LLMResponse) -> Self:
        """Parse the object from an LLM response."""
        ...

    @classmethod
    @abc.abstractmethod
    def to_response_template(cls) -> str:
        """Convert the object to a response template."""
        ...


T_Input = TypeVar('T_Input', bound=LLMFunctionInput)
T_Output = TypeVar('T_Output', bound=LLMFunctionOutput)

T_InputOutput = TypeVar('T_InputOutput') # TODO:

class ChainableFunction(Generic[T_Input, T_Output]):
    def execute(self, obj: T_Input, exception: Optional[Exception] = None) -> T_Output:
        pass


class FunctionChain(Generic[T_Input, T_Output]):
    def __init__(self, functions: Sequence[ChainableFunction]):
        self.functions = functions

    def execute(self, obj: T_Input) -> T_Output:
        # TODO: maybe return a list of outputs; but intermidiate results should be logged
        pass

class LinearFunctionChain(FunctionChain[T_Input, T_Output]):
    """
    FunctionChain that executes functions in a linear order without any error handling.
    Each function should be able to handle errors independently.

    .. mermaid::

        flowchart LR
            A(LLMFunc A) --> B(LLMFunc B)
            A --> A
            B --> B
    """
    def execute(self, obj: T_Input) -> T_Output:
        for function in self.functions:
            obj = function.execute(obj)
        return obj

# do an example for sql generation and execution where limit must be applied
class BacktrackFunctionChain(FunctionChain[T_Input, T_Output]):
    """
    FunctionChain that executes functions in a linear order backtracking errors.
    If a function fails, the chain will backtrack to the previous function and try again.

    .. mermaid::

        flowchart LR
            A(LLMFunc A) --> B(LLMFunc B)
            A --> A
            B --> B
            B --> A
    """
    def __init__(self, functions: Sequence[ChainableFunction], max_backtracks: int = 3):
        super().__init__(functions)
        self.max_backtracks = max_backtracks

    def execute(self, obj: T_Input) -> T_Output:
        index = 0
        results: List[T_InputOutput] = [None] * len(self.functions)
        current_obj: T_InputOutput = obj
        previous_exception: Optional[Exception] = None
        backtrack_count = 0

        while index < len(self.functions):
            try:
                current_obj = self.functions[index].execute(current_obj, previous_exception)
                results.append(current_obj)
                index += 1
            except LLMParsingException as e:
                if index == 0:
                    raise e

                if backtrack_count >= self.max_backtracks:
                    raise LLMParsingException(f"Exceeded maximum backtracks ({self.max_backtracks})")

                backtrack_count += 1
                index -= 1
                previous_exception = e  # propagate error information back;
                # TODO: needs not only the e but current input as well; could be part of e itself

        return results[-1]


class ChainableLLMFunction(ChainableFunction):
    PROMPT_TEMPLATE = """
    {input_obj}

    {response_template}
    """

    # TODO: could extend prompt? But input object should contain enough details
    #  something like {additional_instructions}

    def __init__(self, llm: Union[LLMBase, LLMMiddlewareChain], output_obj_type: Type[T_Output], max_retries: int = 3):
        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._output_obj_type = output_obj_type
        self._context = LLMContext.empty()

        self._max_retries = max_retries

    @classmethod
    def from_config(cls, output_obj_type: Type[T_Output],
                    path_prefix: str = "data", llm_config_path: str = 'llm-config.yaml',
                    max_retries: int = 3) -> ChainableLLMFunction:
        llm = get_llm_from_config(os.path.join(path_prefix, llm_config_path))
        return ChainableLLMFunction(llm, output_obj_type, max_retries)

    def execute(self, obj: T_Input, exception: Optional[Exception] = None) -> T_Output:
        input_prompt = self.PROMPT_TEMPLATE.format(input_obj=obj.to_prompt(),
                                                   response_template=self._output_obj_type.to_response_template())
        messages = [LLMMessage.user_message(input_prompt)]

        retry = 0
        while retry <= self._max_retries:
            request = LLMRequest(context=self._context, messages=messages)
            try:
                llm_response = self._llm_middleware.execute(request)
                return self._output_obj_type.from_response(llm_response)
            except LLMParsingException as e:
                messages.extend([
                    LLMMessage.assistant_message(llm_response.result.first_choice),
                    LLMMessage.user_message(f"Error parsing response: {e}")
                ])

            retry += 1

        raise LLMParsingException("Failed to parse LLM response")


# -------------------------------------------------------------------------------------------------

# How to we do wrapping? Generated SQL -> try execute it and then go back if any problems to regenerate it
#  this will require Exceptions of different types, e.g. levels
#  If I can handle exception of certain level I'll do it but if not it'll buble up to wrapper LLMFunction to handle it.

class DatabaseQuestion(LLMFunctionInput, BaseModel):
    question: str
    table_name: str
    columns_with_types: List[Tuple[str, str]]

    def to_prompt(self) -> str:
        return "\n".join([
            "Database Schema:",
            f"Table: {self.table_name}",
            *[f"{column}: {type}" for column, type in self.columns_with_types],
            "Please generate a SQL query to answer the following question:",
            self.question,
        ])


class SQLQuery(YAMLBlockResponseParser, LLMFunctionOutput, LLMFunctionInput):
    feasible: bool = Field(..., description="Boolean, whether the query is feasible")
    query: str = Field(..., description="SQL query to answer the question OR explanation if the query is not feasible")

    def to_prompt(self) -> str:
        return "\n".join([
            "Analyse the following SQL query and optimize if possible:",
            f"{self.query}"
        ])


class SQLQueryOptimized(YAMLBlockResponseParser, LLMFunctionOutput):
    analysis: str = Field(..., description="Analysis of the query quality and performance")
    optimized_query: str = Field(..., description="Optimized query or original query")


def run_workflow(question: str):
    llm = OpenAILLM.from_env()
    # llm = MockLLM.from_response("\n".join(["```yaml", "feasible: true", "query: SELECT * FROM users", "```"]))
    question_to_query_func: ChainableLLMFunction[DatabaseQuestion, SQLQuery] = ChainableLLMFunction(llm, SQLQuery)

    db_question = DatabaseQuestion(
        question=question,
        table_name="users",
        columns_with_types=[("id", "INTEGER"), ("name", "TEXT"), ("age", "INTEGER")]
    )
    query = question_to_query_func.execute(db_question)
    print(f"question_to_query_func execution result: {type(query)} - {query}")
    if not query.feasible:
        return

    query_to_optimized_query_func: ChainableLLMFunction[SQLQuery, SQLQueryOptimized] = ChainableLLMFunction(llm,
                                                                                                            SQLQueryOptimized)
    optimized_query = query_to_optimized_query_func.execute(query)
    print(f"query_to_optimized_query_func execution result: {type(optimized_query)} - {optimized_query}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    # run_workflow("What is average salary?")
    run_workflow("How many users older than 30 are there?")
