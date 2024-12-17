from typing import List, Tuple

import dotenv
from pydantic import BaseModel, Field

from council import OpenAILLM
from council.llm import YAMLBlockResponseParser
from council.llm.llm_function_chainable import LLMFunctionInput, LLMFunctionOutput, ChainableLLMFunction


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


def test():
    dotenv.load_dotenv()
    run_workflow("What is average salary?")
    # run_workflow("How many users older than 30 are there?")
