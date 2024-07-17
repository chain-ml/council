from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import progressbar  # type: ignore
from council.agents import Agent
from council.contexts import AgentContext, Budget, ScorerContext
from council.scorers import ScorerBase, ScorerException


class AgentTestCaseOutcome(str, Enum):
    Success = "Success"
    Error = "Error"
    Unknown = "Unknown"
    Inconclusive = "Inconclusive"


class ScorerResult:
    def __init__(self, scorer: ScorerBase, score: float) -> None:
        self._scorer = scorer
        self._score = score

    @property
    def scorer(self) -> ScorerBase:
        return self._scorer

    @property
    def score(self) -> float:
        return self._score

    def to_dict(self) -> Dict[str, Any]:
        result = {"score": self._score}
        result.update(self._scorer.to_dict())
        return result


class AgentTestCaseResult:

    def __init__(self, prompt: str, scorers: List[ScorerBase]) -> None:
        self._actual = ""
        self._execution_time = 0.0
        self._outcome = AgentTestCaseOutcome.Unknown
        self._prompt = prompt
        self._scorers: List[ScorerBase] = scorers
        self._scorer_results: List[ScorerResult] = []
        self._error = ""
        self._error_message = ""

    @property
    def outcome(self) -> AgentTestCaseOutcome:
        return self._outcome

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def actual(self) -> str:
        return self._actual

    @property
    def scorer_results(self) -> Sequence[ScorerResult]:
        return self._scorer_results

    @property
    def error(self) -> str:
        return self._error

    @property
    def error_message(self) -> str:
        return self._error_message

    def set_success(self, actual: str, execution_time: float, scores: List[float]) -> None:
        self.set_result(actual, execution_time, scores, AgentTestCaseOutcome.Success)

    def set_error(self, error: Exception, execution_time: float) -> None:
        self.set_result("", execution_time, [], AgentTestCaseOutcome.Error)
        self._error = error.__class__.__name__
        self._error_message = str(error)

    def set_inconclusive(self, execution_time: float) -> None:
        self.set_result("", execution_time, [], AgentTestCaseOutcome.Inconclusive)

    def set_result(
        self, actual: str, execution_time: float, scores: List[float], outcome: AgentTestCaseOutcome
    ) -> None:
        self._actual = actual
        self._execution_time = execution_time
        self._outcome = outcome
        self._scorer_results = [ScorerResult(scorer, score) for (scorer, score) in zip(self._scorers, scores)]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "prompt": self._prompt,
            "message": self._actual,
            "executionTime": self._execution_time,
            "outcome": self._outcome,
            "scorerResults": [item.to_dict() for item in self._scorer_results],
        }
        if self._error != "":
            result["error"] = self._error
        if self._error_message != "":
            result["errorMessage"] = self._error_message
        return result


class AgentTestCase:
    _prompt: str
    _scorers: List[ScorerBase]

    def __init__(self, prompt: str, scorers: List[ScorerBase]) -> None:
        self._prompt = prompt
        self._scorers = scorers

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def scorers(self) -> Sequence[ScorerBase]:
        return self._scorers

    def run(self, agent: Agent) -> AgentTestCaseResult:
        start_time = time.monotonic()

        case_result = AgentTestCaseResult(self._prompt, self._scorers)
        # noinspection PyBroadException
        try:
            context = AgentContext.from_user_message(self._prompt, Budget(10))
            agent_result = agent.execute(context)
        except Exception as e:
            case_result.set_error(e, time.monotonic() - start_time)
            return case_result
        finally:
            duration = time.monotonic() - start_time

        try:
            scores = []
            message = agent_result.try_best_message.unwrap()
            for scorer in self._scorers:
                scores.append(scorer.score(ScorerContext.empty(), message))
            case_result.set_success(message.message, duration, scores)
            return case_result
        except ScorerException:
            case_result.set_inconclusive(duration)
            return case_result


class AgentTestSuiteResult:

    def __init__(self) -> None:
        self._results: List[AgentTestCaseResult] = []

    @property
    def results(self) -> Sequence[AgentTestCaseResult]:
        return self._results

    def add_result(self, result: AgentTestCaseResult) -> None:
        self._results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        results = [item.to_dict() for item in self._results]
        return {"results": results}


class AgentTestSuite:

    def __init__(self, test_cases: Optional[List[AgentTestCase]] = None) -> None:
        self._test_cases: List[AgentTestCase] = test_cases if test_cases is not None else []

    def add_test_case(self, prompt: str, scorers: List[ScorerBase]) -> AgentTestSuite:
        self._test_cases.append(AgentTestCase(prompt, scorers))
        return self

    def run(self, agent: Agent, show_progressbar: bool = True) -> AgentTestSuiteResult:
        pb = None
        if show_progressbar:
            pb = progressbar.ProgressBar(
                maxval=len(self._test_cases),
                widgets=[
                    "Test: ",
                    progressbar.Percentage(),
                    " ",
                    progressbar.Bar(),
                    " ",
                    progressbar.Timer(),
                    " ",
                    progressbar.AdaptiveETA(),
                ],
            ).start()
        try:
            result = AgentTestSuiteResult()
            for index, item in enumerate(self._test_cases):
                case_result = item.run(agent)
                result.add_result(case_result)
                if pb:
                    pb.update(index + 1)
            return result
        finally:
            if pb:
                pb.finish()
