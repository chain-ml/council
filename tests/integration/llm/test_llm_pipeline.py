from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

import dotenv
from pydantic import BaseModel, Field

from council import OpenAILLM
from council.llm import YAMLBlockResponseParser
from council.llm.llm_function.llm_pipeline import (
    LLMProcessor,
    Processor,
    NaivePipelineProcessor,
    BacktrackingPipelineProcessor,
    ProcessorException,
)


class DatabaseQuestion(BaseModel):
    question: str
    table_name: str
    columns_with_types: List[Tuple[str, str]]

    def to_prompt(self) -> str:
        return "\n".join(
            [
                "Database Schema:",
                f"Table: {self.table_name}",
                *[f"{column}: {type}" for column, type in self.columns_with_types],
                "Please generate a SQL query to answer the following question:",
                self.question,
            ]
        )

    @classmethod
    def get_for_test(cls, question: str) -> DatabaseQuestion:
        return DatabaseQuestion(
            question=question,
            table_name="users",
            columns_with_types=[("id", "INTEGER"), ("name", "TEXT"), ("age", "INTEGER")],
        )


class SQLQuery(YAMLBlockResponseParser):
    feasible: bool = Field(..., description="Boolean, whether the query is feasible")
    query: str = Field(..., description="SQL query to answer the question OR explanation if the query is not feasible")

    def to_prompt(self) -> str:
        return "\n".join(["Analyse the following SQL query and optimize if possible:", f"{self.query}"])


class SQLQueryOptimized(YAMLBlockResponseParser):
    analysis: str = Field(..., description="Analysis of the query quality and performance")
    optimized_query: str = Field(..., description="Optimized query or original query")


FakeTable = Dict[str, List[Any]]


class SQLQueryExecutor(Processor[SQLQuery, FakeTable]):
    def execute(self, obj: SQLQuery, exception: Optional[Exception] = None) -> FakeTable:
        if "limit" not in obj.query.lower():
            raise ProcessorException(
                input=obj.to_prompt(), message="Database is huge so query must contain a limit clause"
            )

        return {"id": [1, 2, 3], "name": ["John", "Jane", "Doe"], "age": [28, 34, 29]}


dotenv.load_dotenv()
llm = OpenAILLM.from_env()
question_to_query_proc: LLMProcessor[DatabaseQuestion, SQLQuery] = LLMProcessor(llm, SQLQuery)
query_to_optimized_query_proc: LLMProcessor[SQLQuery, SQLQueryOptimized] = LLMProcessor(llm, SQLQueryOptimized)
execute_query_proc: Processor[SQLQuery, FakeTable] = SQLQueryExecutor()


def test_question_to_query_proc() -> None:
    db_question = DatabaseQuestion.get_for_test("What is average salary?")
    query = question_to_query_proc.execute(db_question)
    assert isinstance(query, SQLQuery)
    assert not query.feasible


def test_question_to_optimized_query_proc() -> None:
    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    query = question_to_query_proc.execute(db_question)
    assert isinstance(query, SQLQuery)
    assert query.feasible

    optimized_query = query_to_optimized_query_proc.execute(query)
    assert isinstance(optimized_query, SQLQueryOptimized)


def test_naive_pipeline() -> None:
    pipeline: NaivePipelineProcessor[DatabaseQuestion, SQLQueryOptimized] = NaivePipelineProcessor(
        [question_to_query_proc, query_to_optimized_query_proc]
    )

    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    optimized_query = pipeline.execute(db_question)
    assert isinstance(optimized_query, SQLQueryOptimized)
    # assert on records length


def test_backtracking_pipeline() -> None:
    pipeline: BacktrackingPipelineProcessor[DatabaseQuestion, FakeTable] = BacktrackingPipelineProcessor(
        [question_to_query_proc, execute_query_proc]
    )

    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    result = pipeline.execute(db_question)
    print(result)


# add test case for query -> optimized_query -> execution with transfer_to
