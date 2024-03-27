"""

A module for running test cases on agent instances to evaluate their performance with custom prompts and scorers.

This module includes functionalities to define test cases for agents, to execute these test cases, and to accumulate
the results. A test case is based on a given prompt and associated scorers, which quantify the performance of an agent.
The module allows the definition of individual test cases, as well as bundling multiple cases into a test suite to be run against an agent.

Classes:
    AgentTestCaseOutcome - An enumeration of possible outcomes for a test case.
    ScorerResult - Contains the results of scoring an agent's response.
    AgentTestCaseResult - Stores the outcome, execution time, and other metadata of a run test case.
    AgentTestCase - Defines a single test case with a prompt and associated scorers.
    AgentTestSuiteResult - Accumulates the results of all test cases within a test suite.
    AgentTestSuite - Contains a collection of test cases to be run as a test suite against an agent.


"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Dict, Any, Sequence, Optional
import progressbar  # type: ignore

from council.agents import Agent
from council.scorers import ScorerBase, ScorerException
from council.contexts import (
    AgentContext,
    Budget,
    ScorerContext,
)


class AgentTestCaseOutcome(str, Enum):
    """
    An enumeration to represent the possible outcomes of a test case executed against an agent.
    This class inherits from the `str` data type and the `Enum` class of the enum module. It is
    used to define a set of named constants that represent the status of an agent's test case outcome.
    
    Attributes:
        Success (str):
             Represents a successful outcome where the agent has passed the test case.
        Error (str):
             Represents an outcome where the agent has encountered an error during the test case.
        Unknown (str):
             Represents an outcome where the result of the test case is unknown.
        Inconclusive (str):
             Represents an outcome where the test case did not conclusively pass or fail.
        

    """
    Success = "Success"
    Error = "Error"
    Unknown = "Unknown"
    Inconclusive = "Inconclusive"


class ScorerResult:
    """
    A class used to encapsulate the result of a scoring operation including the scorer instance and the computed score.
    
    Attributes:
        _scorer (ScorerBase):
             The scorer object used to compute the score. This should be an instance of a class derived from ScorerBase.
        _score (float):
             The numerical value of the score computed by the scorer.
    
    Methods:
        __init__(self, scorer:
             ScorerBase, score: float):
            Initializes the ScorerResult instance with a scorer object and score value.
    
    Args:
        scorer (ScorerBase):
             The scorer object used for the computation of the score.
        score (float):
             The computed score as a numerical value.
            @property
        scorer(self) -> ScorerBase:
            The property that returns the scorer object.
    
    Returns:
        (ScorerBase):
             The scorer object used to compute the score.
            @property
        score(self) -> float:
            The property that returns the score value.
    
    Returns:
        (float):
             The numerical score value computed by the scorer.
        to_dict(self) -> Dict[str, Any]:
            Converts the ScorerResult's information into a dictionary, including the score and the scorer's details.
    
    Returns:
        (Dict[str, Any]):
             A dictionary containing the score and the scorer's serializable information.

    """
    def __init__(self, scorer: ScorerBase, score: float):
        """
        Initialize a new instance with a scorer object and a score value.
        
        Args:
            scorer (ScorerBase):
                 An instance of a scorer object that contains the mechanism
                to score a particular set of data.
            score (float):
                 The score value obtained from the scorer object for a
                particular dataset.
            

        """
        self._scorer = scorer
        self._score = score

    @property
    def scorer(self) -> ScorerBase:
        """
        Accessor method for the `_scorer` attribute, which should be an instance of ScorerBase or one of its subclasses.
        This method allows retrieval of the object set for scoring in the encompassing class. It acts as a property,
        and it should be accessed as an attribute without the need for calling it as a function.
        
        Returns:
            (ScorerBase):
                 An instance of ScorerBase or a subclass thereof which is used to calculate
                scores based on certain criteria or algorithms.

        """
        return self._scorer

    @property
    def score(self) -> float:
        """
        
        Returns the score property of the instance.
            This method is a property that when accessed, returns the internal
            _score attribute representing some form of a score (e.g., a test score,
            game score, evaluation score). Since this is a property, it should be accessed
            like an attribute without calling it like a method.
        
        Returns:
            (float):
                 The current value of the _score attribute.
            

        """
        return self._score

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object properties into a dictionary.
        This method extracts the score and the scorer details from the object, and constructs a dictionary with
        this information. It starts with a dictionary containing the score attribute, then updates this
        dictionary with the contents of the object's scorer's own to_dict method.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representation of the object, including the score and
                the scorer properties.
            

        """
        result = {"score": self._score}
        result.update(self._scorer.to_dict())
        return result


class AgentTestCaseResult:
    """
    A class that encapsulates the result data of a test case for an agent, including the actual result, execution time,
    outcome, prompt, scorers, scorer results, and error information if any exists.
    
    Attributes:
        _actual (str):
             The actual result of the test case.
        _execution_time (float):
             The time taken to execute the test case.
        _outcome (AgentTestCaseOutcome):
             The outcome of the test case, typically Success, Error, or Inconclusive.
        _prompt (str):
             The prompt or input given to the agent for the test case.
        _scorers (List[ScorerBase]):
             The list of scorers used to evaluate the test case.
        _scorer_results (List[ScorerResult]):
             The results from each scorer after evaluating the test case.
        _error (str):
             The type of error that occurred, if any.
        _error_message (str):
             The error message associated with an error, if any.
    
    Methods:
        __init__(self, prompt:
             str, scorers: List[ScorerBase]): Initializes a new instance of AgentTestCaseResult with the given prompt and scorers.
        outcome(self):
             The property that gets the outcome of the test case.
        prompt(self):
             The property that gets the prompt used for the test case.
        actual(self):
             The property that gets the actual result of the test case.
        scorer_results(self):
             The property that gets the results from each scorer.
        error(self):
             The property that gets the type of error that occurred, if any.
        error_message(self):
             The property that gets the error message associated with an error.
        set_success(self, actual:
             str, execution_time: float, scores: List[float]): Records a successful test case result.
        set_error(self, error:
             Exception, execution_time: float): Records an error that occurred during the test case execution.
        set_inconclusive(self, execution_time:
             float): Records an inconclusive test case result.
        set_result(self, actual:
             str, execution_time: float, scores: List[float], outcome: AgentTestCaseOutcome): Sets the actual result, execution time, scores, and outcome of the test case.
        to_dict(self):
             Converts the test case result data into a dictionary format.

    """
    _actual: str
    _execution_time: float
    _outcome: AgentTestCaseOutcome
    _prompt: str
    _scorers: List[ScorerBase]
    _scorer_results: List[ScorerResult]
    _error: str
    _error_message: str

    def __init__(self, prompt: str, scorers: List[ScorerBase]):
        """
        Initializes a new instance of the class with the given prompt and list of scorers.
        This constructor sets up instance variables to store the prompt, scorers, and initial values for various
        status indicators such as execution time, outcome, scorer results, and error information.
        
        Args:
            prompt (str):
                 The input prompt for which the scorers are to be applied.
            scorers (List[ScorerBase]):
                 A list of scorer instances derived from the ScorerBase class used to evaluate the prompt.
        
        Attributes:
            _actual (str):
                 The actual response generated, initialized to an empty string.
            _execution_time (int):
                 The time taken to execute the scorers, initialized to 0.
            _outcome (AgentTestCaseOutcome):
                 The outcome of the test case, initially set to Unknown.
            _prompt (str):
                 The input prompt for which the test case is constructed.
            _scorers (List[ScorerBase]):
                 The list of scorers that will be applied to the prompt.
            _scorer_results (list):
                 A list to store the results from each scorer, initially an empty list.
            _error (str):
                 A string to store error details, initialized to an empty string.
            _error_message (str):
                 A string to store the error message, if any, initialized to an empty string.

        """
        self._actual = ""
        self._execution_time = 0
        self._outcome = AgentTestCaseOutcome.Unknown
        self._prompt = prompt
        self._scorers = scorers
        self._scorer_results = []
        self._error = ""
        self._error_message = ""

    @property
    def outcome(self) -> AgentTestCaseOutcome:
        """
        Property that gets the test case outcome of an agent.
        
        Returns:
            (AgentTestCaseOutcome):
                 An enum value representing the outcome of the test case.
                This read-only property allows access to the internal _outcome attribute, which holds the result of a test case that has been run on the agent. The outcome is represented by an enumeration, which could define values such as PASS, FAIL, ERROR, or SKIP, indicating the state of a test after execution.

        """
        return self._outcome

    @property
    def prompt(self) -> str:
        """
        Gets the prompt value of the instance.
        
        Returns:
            (str):
                 The current prompt value stored in the instance.

        """
        return self._prompt

    @property
    def actual(self) -> str:
        """
        Retrieves the actual value of the object.
        This is a property decorator that allows for accessing the `_actual` attribute of the object,
        which represents the actual value it holds, in a controlled manner. The property is read-only.
        
        Returns:
            (str):
                 The actual value stored within the `_actual` attribute.
            

        """
        return self._actual

    @property
    def scorer_results(self) -> Sequence[ScorerResult]:
        """
        Getter for the `_scorer_results` attribute of the object instance.
        
        Returns:
            (Sequence[ScorerResult]):
                 A sequence containing the results of the scoring process, typically
                with elements being instances of a `ScorerResult` or similar data structure.

        """
        return self._scorer_results

    @property
    def error(self) -> str:
        """
        Property that gets the current error message.
        
        Returns:
            (str):
                 The error message stored in the '_error' attribute.

        """
        return self._error

    @property
    def error_message(self) -> str:
        """
        Gets the error message for the object instance.
        
        Returns:
            (str):
                 The error message stored in the object instance.

        """
        return self._error_message

    def set_success(self, actual: str, execution_time: float, scores: List[float]):
        """
        Sets the test case result with a successful outcome.
        This method registers the result of a test case, marking it as successful. It records the actual output, the execution time, and the scores associated with the test case execution.
        
        Args:
            actual (str):
                 The actual output produced by the test case execution.
            execution_time (float):
                 The time taken for the test case to execute.
            scores (List[float]):
                 A list of scores representing the assessment of the test case results.
        
        Returns:
            None

        """
        self.set_result(actual, execution_time, scores, AgentTestCaseOutcome.Success)

    def set_error(self, error: Exception, execution_time: float):
        """
        Sets the error information for the current test case, along with the execution time of the run that encountered the error.
        This function records the exception details into the instance by capturing the name of the error class and the string representation of the exception. Additionally, it sets the result of the test case to indicate an error with an empty response, and logs the execution time of the failing run.
        
        Args:
            error (Exception):
                 The exception object that represents the error encountered during test case execution.
            execution_time (float):
                 The time in seconds that it took to encounter the error during the execution of the test case.
                No return value as this function alters the state of the instance to record the error and execution time.

        """
        self.set_result("", execution_time, [], AgentTestCaseOutcome.Error)
        self._error = error.__class__.__name__
        self._error_message = str(error)

    def set_inconclusive(self, execution_time: float):
        """
        Sets the test case result as inconclusive.
        This method updates the result of the test case to represent an inconclusive result, indicating that the test neither passed nor failed due to insufficient information or an unresolvable state during execution.
        
        Args:
            execution_time (float):
                 The duration taken for the test case execution in seconds.
            

        """
        self.set_result("", execution_time, [], AgentTestCaseOutcome.Inconclusive)

    def set_result(self, actual: str, execution_time: float, scores: List[float], outcome: AgentTestCaseOutcome):
        """
        Sets the result information for a test case execution, including the actual output, execution time, list of scores, and outcome of the test case.
        
        Args:
            actual (str):
                 The actual output produced by the execution of the test case
            execution_time (float):
                 The time taken to execute the test case
            scores (List[float]):
                 A list of scores obtained by applying different scoring methods
            outcome (AgentTestCaseOutcome):
                 The outcome of the test case (e.g., pass, fail, etc.)
            Side effects:
                - Directly updates the instance attributes _actual, _execution_time, _outcome, and _scorer_results
                based on the provided arguments.
                - Initializes the list of ScorerResult objects based on provided scorers and their corresponding scores.

        """
        self._actual = actual
        self._execution_time = execution_time
        self._outcome = outcome
        self._scorer_results = [ScorerResult(scorer, score) for (scorer, score) in zip(self._scorers, scores)]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object into a dictionary representation.
        This method creates a dictionary with the object's prompt, message, executionTime, outcome, and
        scorerResults fields. If there's an error or an error message present, these fields are also included in the result.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representing the key aspects of the object. The dictionary
            (includes the following fields):
            (- 'prompt'):
                 The original prompt given to the object.
            (- 'message'):
                 The actual message/response from the object.
            (- 'executionTime'):
                 The total time taken for execution.
            (- 'outcome'):
                 The result or outcome of the execution.
            (- 'scorerResults'):
                 A list of dictionaries containing scorer results, each converted by
                calling their respective 'to_dict' method.
            - 'error' (optional):
                 An error field included if there was an error during the process.
            - 'errorMessage' (optional):
                 A message describing the error, if applicable.
            

        """
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
    """
    A class to encapsulate and execute test cases for an agent, with the ability to evaluate agent responses using a collection of scorers.
    
    Attributes:
        _prompt (str):
             The prompt or user message used to test the agent.
        _scorers (List[ScorerBase]):
             A list of ScorerBase objects used to score the agent's responses.
    
    Args:
        prompt (str):
             The test case's initial user message to the agent.
        scorers (List[ScorerBase]):
             A sequence of scorers that will assess the quality of the agent's responses.
    
    Methods:
        prompt (Property[str]):
             Retrieves the test prompt.
        scorers (Property[Sequence[ScorerBase]]):
             Retrieves the sequence of scorers.
        run(agent:
             Agent) -> AgentTestCaseResult:
            Executes the test case using the given agent, collecting the result and any associated scores from the defined scorers. If an exception occurs during agent execution, it captures the error. If an exception occurs during scoring, the test case result is set to inconclusive.
    
    Args:
        agent (Agent):
             The agent instance that will process the prompt and generate a response.
    
    Returns:
        (AgentTestCaseResult):
             An object encapsulating the outcome of the test case, including the agent's response,
            the scoring results, and any error or exception information.
    
    Raises:
        ScorerException:
             If any scorer encounters an issue while scoring the agent's response, the test case result is marked as inconclusive.

    """
    _prompt: str
    _scorers: List[ScorerBase]

    def __init__(self, prompt: str, scorers: List[ScorerBase]):
        """
        Initializes the instance with a prompt and a list of scorer instances.
        
        Args:
            prompt (str):
                 The text prompt to which the scorers will be applied.
            scorers (List[ScorerBase]):
                 A list of scorer instances that are subclasses of ScorerBase.
        
        Raises:
            TypeError:
                 If the scorers are not instances of ScorerBase. This is not explicitly included in the function body, but it's a possible error if type checking is enforced.
            

        """
        self._prompt = prompt
        self._scorers = scorers

    @property
    def prompt(self) -> str:
        """
        Property that gets the current prompt value.
        This property is used to retrieve the current prompt value that has been set for an instance.
        The value is stored in a private variable _prompt.
        
        Returns:
            (str):
                 The current prompt string.

        """
        return self._prompt

    @property
    def scorers(self) -> Sequence[ScorerBase]:
        """
        Gets the list of scorer objects used in the evaluation process.
        This property returns the list of ScorerBase instances that are being
        used as part of the evaluation mechanism for a model or algorithm.
        Each scorer in the list represents a different metric or criteria used
        for scoring.
        
        Returns:
            (Sequence[ScorerBase]):
                 A sequence (such as a list or tuple) of ScorerBase
                objects that are currently in use.

        """
        return self._scorers

    def run(self, agent: Agent) -> AgentTestCaseResult:
        """
        Runs a test case using the provided agent and records the results.
        This method executes a test case with a given prompt, scoring the result
        using the specified scorer or scorers. It tracks the duration of the test run
        and stores the outcome, actual responses, scorer results, errors, and error messages.
        It caters for multiple scenarios including standard execution, error occurrences,
        and inconclusive results (e.g., due to exceptions raised during scoring).
        
        Args:
            agent (Agent):
                 The agent object that will execute the test case.
        
        Returns:
            (AgentTestCaseResult):
                 An object containing the results of the test case run.
                This includes information on the actual output from the agent, execution time,
                computed scores, and potential errors occurred during execution.
        
        Raises:
            ScorerException:
                 An exception raised if there is an issue with scoring the response from the agent.

        """
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
    """
    A class to collect and store results of tests performed on Agent test cases.
    This class encapsulates a list to maintain test results and provides methods to add a test result and to convert results into a dictionary.
    
    Attributes:
        _results (List[AgentTestCaseResult]):
             A private list that holds the results of the Agent test cases.
    
    Methods:
        __init__(self):
            Initializes a new instance of AgentTestSuiteResult with an empty list of test case results.
        results(self) -> Sequence[AgentTestCaseResult]:
            Gets the stored test case results. This is a @property decorated method that allows
            the underscore-prefixed _results attribute to be accessed in a read-only fashion.
        add_result(self, result:
             AgentTestCaseResult) -> None:
            Appends a test case result to the internal list of results.
    
    Args:
        result (AgentTestCaseResult):
             An instance of AgentTestCaseResult to be
            added to the collection of results.
        to_dict(self) -> Dict[str, Any]:
            Converts the stored test case results into a dictionary format.
    
    Returns:
        (Dict[str, Any]):
             A dictionary with a key 'results' where the value is a list
            of dictionaries representing individual test case results.

    """
    _results: List[AgentTestCaseResult]

    def __init__(self):
        """
        Initializes a new instance of the class, setting up an empty list to store results.
        
        Attributes:
            _results (list):
                 A private list that will hold the results for further processing or usage.
            

        """
        self._results = []

    @property
    def results(self) -> Sequence[AgentTestCaseResult]:
        """
        
        Returns the sequence of test case results for the agents.
            This property function provides access to the internal sequence of
            AgentTestCaseResult objects. These objects contain the results of specific
            test cases that have been run on agents. The sequence includes results from
            all the test cases that have been executed up to the point of access.
        
        Returns:
            (Sequence[AgentTestCaseResult]):
                 A sequence of AgentTestCaseResult objects
                representing the results of each test case run on agents.

        """
        return self._results

    def add_result(self, result: AgentTestCaseResult) -> None:
        """
        Adds a test case result to the result list of the object.
        
        Args:
            result (AgentTestCaseResult):
                 An instance of AgentTestCaseResult to be added to the results list held by this object.
        
        Returns:
            (None):
                 This method does not return anything. It just appends the result to the internal results list.

        """
        self._results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the internal state of an object's results into a dictionary format.
        This method iterates over each result item in the object's '_results' attribute,
        calling the 'to_dict' method of each item, and collects the resulting dictionaries into
        a list. It then returns a dictionary with a single key 'results' paired with this list.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing a key 'results' with a list of dictionaries
                as its value, each dictionary being the converted form of an item in '_results'.
            

        """
        results = []
        for item in self._results:
            results.append(item.to_dict())
        result = {"results": results}
        return result


class AgentTestSuite:
    """
    A suite for aggregating and running a collection of AgentTestCase instances on a given Agent.
    
    Attributes:
        _test_cases (List[AgentTestCase]):
             A private list that holds all the test cases to be run.
    
    Args:
        test_cases (Optional[List[AgentTestCase]], optional):
             An optional list of `AgentTestCase` objects
            for pre-populating the test suite. If None, a new, empty list is initialized.
    
    Methods:
        __init__:
             Constructs an instance of `AgentTestSuite` with optional test cases.
        add_test_case:
             Adds a new test case to the suite's test case list.
        run:
             Executes all test cases in the suite against a given agent and returns the results.
    
    Raises:
        Exception:
             If any occurs during the test case execution within the `run` method.

    """
    _test_cases: List[AgentTestCase]

    def __init__(self, test_cases: Optional[List[AgentTestCase]] = None):
        """
        Initializes the instance with a list of test cases.
        This constructor method optionally takes a list of test cases to be assigned to the instance. If no list is
        provided, it initializes an empty list of test cases.
        
        Args:
            test_cases (Optional[List[AgentTestCase]]):
                 A list of AgentTestCase objects to be used for testing.
                If None, initializes to an empty list.
            

        """
        if test_cases is not None:
            self._test_cases = test_cases
        else:
            self._test_cases = []

    def add_test_case(self, prompt: str, scorers: List[ScorerBase]) -> AgentTestSuite:
        """
        Adds a new test case to the test suite for an agent.
        
        Args:
            prompt (str):
                 The input prompt or question that will be presented to the agent.
            scorers (List[ScorerBase]):
                 A list of scorer instances that will be used to evaluate the agent's response to the prompt.
        
        Returns:
            (AgentTestSuite):
                 The instance of the test suite with the newly added test case.
        
        Raises:
            TypeError:
                 If the `scorers` argument is not a list of ScorerBase instances.

        """
        self._test_cases.append(AgentTestCase(prompt, scorers))
        return self

    def run(self, agent: Agent, show_progressbar: bool = True) -> AgentTestSuiteResult:
        """
        Executes all test cases on the provided agent and collects their results.
        This method iterates over the test cases defined within the suite, runs each one
        against the provided agent, and appends the individual test case results into an
        `AgentTestSuiteResult` object. If `show_progressbar` is set to `True`, it displays
        a progress bar in the console to indicate the progress of the test execution.
        
        Args:
            agent (Agent):
                 The agent instance on which the test cases will be run.
            show_progressbar (bool, optional):
                 Flag to determine if a progress bar should be
                shown in the console. Defaults to True.
        
        Returns:
            (AgentTestSuiteResult):
                 An object containing the results of all test cases.
        
        Raises:
            Any exceptions raised during the execution of test cases will propagate up to the
            caller, as no exception handling is performed within this method.

        """
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
