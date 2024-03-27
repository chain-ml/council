"""

A module that provides mock implementations for various components involved in a conversational AI framework, primarily for testing purposes.

This module includes the following key classes which simulate different behaviors for testing:

- MockMultipleResponses: Simulates an LLM-style function that returns predetermined responses to a list of messages.

- LLMMessagesToStr (Protocol): An interface that requires the implementation of a method to convert a sequence of LLMMessage objects into a sequence of strings.

- llm_message_content_to_str: Utility function that extracts the content of each LLMMessage in a sequence and returns a matching sequence of strings.

- MockTokenCounter: Counts the tokens of a sequence of LLMMessage objects and optionally raises an exception if the count exceeds a specified limit.

- MockSkill: Mimics a basic skill with customizable action behavior, primarily for testing SkillBase functionalities.

- MockLLM: Provides a mock LLM (Language Learning Model) class that can simulate the behavior of LLM instances by returning predefined strings as results to given messages.

- MockErrorLLM: Simulates an LLM that always raises a specified exception when handling chat requests.

- MockErrorSimilarityScorer: Simulates a scoring component that raises an exception during the scoring process.

- MockAgent: A mock agent that simulates the generation of agent messages and the randomness in response latency.

- MockErrorAgent: An agent that raises an exception when executing its function.

- MockMonitored: Creates a mock monitored entity for testing monitorable behavior and interface.

The above components are useful for setting up test cases to ensure different parts of the conversational AI system interact correctly and handle various scenarios, including normal operations, error handling, and token limitations.


"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence

from council.agents import Agent, AgentResult
from council.contexts import (
    AgentContext,
    Budget,
    ChatMessage,
    LLMContext,
    Monitorable,
    Monitored,
    ScoredChatMessage,
    ScorerContext,
    SkillContext,
)
from council.llm import LLMBase, LLMException, LLMMessage, LLMResult, LLMTokenLimitException, LLMessageTokenCounterBase
from council.scorers import ScorerBase
from council.skills import SkillBase


class MockMultipleResponses:
    """
    A class for simulating multiple mock responses for testing purposes.
    This class is meant to simulate a sequence of responses from a list of
    predefined responses. It is initialized with a nested list where each
    inner list represents a separate response split into lines. The class
    is callable and, upon each call, it returns the next response from the
    sequence until all responses have been used. The responses are cycled
    through in the order they were provided during initialization.
    
    Attributes:
        _count (int):
             A counter to keep track of the number of times the
            instance has been called and to determine which mock response to
            provide next.
        _responses (List[str]):
             A list of mock responses, where each response
            has been formed by joining the lines of the corresponding inner
            list passed during instantiation.
    
    Methods:
        __call__(messages:
             Sequence[LLMMessage]) -> Sequence[str]: A method
            allowing the instance to be called like a function. Delegates to
            `call` method to return a mock response.
        call(_messages:
             Sequence[LLMMessage]) -> Sequence[str]: Returns the
            next mock response from the sequence based on the current count.
            This increments the count with each call, unless the end of the
            response sequence has been reached.
    
    Args:
        responses (List[List[str]]):
             A nested list where each inner list
            contains parts of the mock response, expected to be joined into
            a single string.
    
    Raises:
        IndexError:
             If the call method is invoked more times than there are
            prepared responses in the '_responses' list.

    """
    def __init__(self, responses: List[List[str]]):
        """
        Initializes a new instance of the class with a list of responses.
        This method takes a list of list of strings, where each inner list represents
        a multi-line response. It joins each inner list into a single string, separated by newlines.
        The result is then stored in the instance variable `_responses`. It also initializes
        a counter `_count` which is used to keep track of something (not specified here).
        
        Args:
            responses (List[List[str]]):
                 A list of responses where each response is a list of strings.
            

        """
        self._count = 0
        self._responses = ["\n".join(resp) for resp in responses]

    def __call__(self, messages: Sequence[LLMMessage]) -> Sequence[str]:
        """
        This method is a callable interface that delegates the call to the instance's `call` method.
        
        It processes a sequence of `LLMMessage` objects and returns a sequence of strings as the output.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of message objects that need to be processed.
            
        
        Returns:
            (Sequence[str]):
                 The processed output as a sequence of strings corresponding to the input messages.
            

        """
        return self.call(messages)

    def call(self, _messages: Sequence[LLMMessage]) -> Sequence[str]:
        """
        Gets the response from a fixed list for each call based on the sequence of messages provided.
        
        Args:
            _messages (Sequence[LLMMessage]):
                 A sequence of messages passed to the function.
        
        Returns:
            (Sequence[str]):
                 A list containing the response string. The response is the element from
                the fixed list of responses associated with the current count. With each call, the count
                is incremented to provide the next response in a subsequent call.
        
        Raises:
            IndexError:
                 If the _count exceeds the number of preset responses.

        """
        if self._count < len(self._responses):
            self._count += 1
        return [self._responses[self._count - 1]]


class LLMMessagesToStr(Protocol):
    """
    A protocol for converting a sequence of LLMMessage instances to a sequence of strings.
    This protocol defines a single method '__call__' that, when implemented, should take a sequence of
    LLMMessage objects and return a sequence of strings corresponding to those messages.

    Attributes:
        No attributes are defined for this protocol.

    Methods:
        __call__(self, messages:
             Sequence[LLMMessage]) -> Sequence[str]:
            This method should be implemented to define the conversion of LLMMessage instances to strings.

    Args:
        messages:
             A sequence of LLMMessage instances to be converted.

    Returns:
        A sequence of strings, each corresponding to an LLMMessage from the input sequence.


    """

    def __call__(self, messages: Sequence[LLMMessage]) -> Sequence[str]:
        """
       Calls the instance as a function and processes a sequence of messages to generate responses.

       Args:
           messages (Sequence[LLMMessage]):
                A sequence of LLMMessage objects to be processed.

       Returns:
           (Sequence[str]):
                A sequence of response strings corresponding to each input message.

       """
        ...


def llm_message_content_to_str(messages: Sequence[LLMMessage]) -> Sequence[str]:
    """
    Converts a sequence of LLMMessage objects to a sequence of their string content.

    """
    return [msg.content for msg in messages]


class MockTokenCounter(LLMessageTokenCounterBase):
    """
    A mock implementation of `LLMessageTokenCounterBase` designed for counting the tokens in a sequence of `LLMMessage` objects, with an optional limit on the number of tokens.
    
    Attributes:
        _limit (int):
             An optional upper limit on the token count. If the count exceeds the limit, a `LLMTokenLimitException` is raised. The default is -1, which means there is no limit.
    
    Methods:
        __init__(limit:
             int = -1):
            Initializes a new instance of `MockTokenCounter` with an optional token count limit.
        count_messages_token(messages:
             Sequence[LLMMessage]) -> int:
            Counts the number of tokens in a sequence of `LLMMessage` instances.
    
    Args:
        messages (Sequence[LLMMessage]):
             A sequence of messages for which to count tokens.
    
    Returns:
        (int):
             The total count of tokens in the messages.
    
    Raises:
        LLMTokenLimitException:
             If the total token count exceeds the specified limit.

    """
    def __init__(self, limit: int = -1):
        """
        Initializes an instance of the class with an optional limit.
        This constructor sets the limit for an instance which may control the scope,
        size, or extent of the instance's operation or responsibility.
        
        Args:
            limit (int, optional):
                 An upper bound to the effect or operation of the instance.
                Provides a non-restrictive, limitless default when set to -1.
        
        Attributes:
            _limit (int):
                 A private variable that holds the upper bound.
            

        """
        self._limit = limit

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        """
        Counts the tokens in a sequence of LLMMessage objects and returns the total token count.
        If the total token count exceeds the predefined limit, it raises an LLMTokenLimitException.
        This function iterates over the provided messages, sums up the length of their content attributes, and ensures the aggregate token count does not surpass the set token limit.
        
        Args:
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage objects whose tokens are to be counted.
        
        Returns:
            (int):
                 The total count of tokens in the input messages.
        
        Raises:
            LLMTokenLimitException:
                 If the total token count exceeds the set limit.

        """
        result = 0
        for msg in messages:
            result += len(msg.content)
            if 0 < self._limit < result:
                raise LLMTokenLimitException(
                    token_count=result, limit=self._limit, model="mock", llm_name=f"{self.__class__.__name__}"
                )
        return result


class MockSkill(SkillBase):
    """
    A mock skill class serving as a stub for testing purposes, inheriting from SkillBase class. This class allows the simulation of skill execution with custom actions and responses.
    
    Attributes:
        _action (Callable[[SkillContext], ChatMessage]):
             An optional callable that takes a SkillContext and returns a ChatMessage. Defaults to creating an empty message.
    
    Methods:
        __init__:
             Constructor for the MockSkill class. Optionally sets the skill's action to a given callable.
        execute:
             Triggers the execution of the skill's action and returns a ChatMessage.
        empty_message:
             Generates an empty success message to act as a default response.
        set_action_custom_message:
             A setter that changes the current action to return a custom success message.
        build_wait_skill:
             A static method that constructs a new MockSkill that simulates a delay before responding.
    
    Args:
        name (str):
             The name of the skill, with a default value of 'mock'.
        action (Optional[Callable[[SkillContext], ChatMessage]]):
             An optional callable to define the skill's behavior.
    
    Returns:
        (MockSkill):
             An instance of MockSkill with the specified action.

    """
    def __init__(self, name: str = "mock", action: Optional[Callable[[SkillContext], ChatMessage]] = None):
        """
        Initializes a new instance of the class.
        This constructor initializes the object with a specified name and an optional action. If the action is not provided, it defaults to a predefined empty_message method.
        
        Args:
            name (str):
                 The name of the instance, defaults to 'mock' if not provided.
            action (Optional[Callable[[SkillContext], ChatMessage]]):
                 An optional callable that takes a SkillContext and returns a ChatMessage. It represents the action that will be performed, defaults to empty_message method if None.
            

        """
        super().__init__(name)
        self._action = action if action is not None else self.empty_message

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes an action within a given context and returns the resultant chat message.
        
        Args:
            context (SkillContext):
                 An instance of SkillContext which encapsulates the context
                within which the action should be executed.
        
        Returns:
            (ChatMessage):
                 The message that results from the action execution.
        
        Raises:
        
        Raises an exception if the action cannot be executed or the context is invalid. The
            specific type of exception depends on the underlying action logic.

        """
        return self._action(context)

    def empty_message(self, context: SkillContext):
        """
        Determines the default rank for the execution unit based on parallelism support.
        This property method evaluates the presence of parallelism within the execution context and
        provides a default rank accordingly. If parallelism is enabled, it returns `1` suggesting
        that the execution unit should be ranked at the top level. If parallelism is not enabled,
        it returns `None` indicating that no rank is applicable.
        
        Returns:
            (Optional[int]):
                 The default rank `1` if parallelism is supported, otherwise `None`.

        """
        return self.build_success_message("")

    def set_action_custom_message(self, message: str) -> None:
        """
        Sets a custom success message for an action within the object.
        This function sets the `_action` attribute to a lambda function that when called, generates a success message using the provided `message` string. Typically, this method would be used to define what success looks like for an action that the object will perform, encapsulating this within a custom message.
        
        Args:
            message (str):
                 The custom message to be used as the success message for the action.
        
        Returns:
            (None):
                 This method does not return anything.

        """
        self._action = lambda context: self.build_success_message(message)

    @staticmethod
    def build_wait_skill(duration: int = 1, message: str = "done") -> MockSkill:
        """
        Builds and returns a MockSkill object configured to wait a specific duration before sending a chat message.
        
        Args:
            duration (int):
                 The number of seconds the skill should wait before sending a chat message. Default is 1.
            message (str):
                 The chat message to be sent after the duration. Default is 'done'.
        
        Returns:
            (MockSkill):
                 An instance of MockSkill configured to perform the waiting action with the specified message.

        """
        def wait_a_message(context: SkillContext) -> ChatMessage:
            """
            Waits for a specific duration before returning a ChatMessage instance.
            This function takes a `SkillContext` object as input, pauses execution of the code for an amount of time
            defined inside the `SkillContext`, and afterwards, creates and returns a `ChatMessage` object constructed
            with a message also obtained from the `SkillContext`.
            
            Parameters:
                context (SkillContext):
                     The context from which the sleep duration and message are obtained.
            
            Returns:
                (ChatMessage):
                     An instance of `ChatMessage` containing the message from the given `SkillContext`.
            
            Raises:
                AttributeError:
                     If the `duration` or `message` attributes are not found in the `SkillContext` object.
                ValueError:
                     If the `duration` attribute is not a positive number.
                

            """
            time.sleep(duration)
            return ChatMessage.skill(message)

        if duration > 0:
            return MockSkill(action=wait_a_message)
        return MockSkill(action=lambda context: ChatMessage.skill(message))


class MockLLM(LLMBase):
    """
    A mock subclass of LLMBase for testing purposes that mimics the behaviour of a language model without making actual API calls.
    This class is designed to be used for testing by providing predetermined responses to messages sent to the model. It allows
    for the customization of responses based on the input messages and can be useful in contexts where the testing does not
    require interaction with a real language model.
    
    Attributes:
        _action (Optional[LLMMessagesToStr]):
             A callable that defines how the model responds to input messages. Can be None.
        token_limit (int):
             An integer specifying the token limit for the mock token counter. The default value of -1 indicates
            no limit.
    
    Methods:
        __init__:
             Initializes a new instance of the MockLLM class.
        _post_chat_request:
             Simulates sending a chat request to the language model and returns a mock result.
        from_responses:
             Creates a new instance of MockLLM that will generate fixed responses from a list of strings.
        from_response:
             Creates a new instance of MockLLM that will always respond with the same string.
        from_multi_line_response:
             Creates a new instance of MockLLM that will respond with a single string formed by
            joining multiple strings separated by a newline character.
            The class provides utility methods for creating instances of MockLLM that return static or dynamic responses to input.
            These can be particularly useful for unit tests or any scenario requiring predictable outcomes without the need for a
            live model.

    """
    def __init__(self, action: Optional[LLMMessagesToStr] = None, token_limit: int = -1):
        """
        Initializes a new instance of the class.
        
        Args:
            action (Optional[LLMMessagesToStr]):
                 An optional callable that converts LLMMessages to a string representation.
                Default is None, which means no conversion will be applied.
            token_limit (int):
                 The maximum number of tokens allowed. If set to -1, there will be no limit
                on the number of tokens. This is used to instantiate a MockTokenCounter with the given
                token limit as its threshold.

        """
        super().__init__(token_counter=MockTokenCounter(token_limit))
        self._action = action

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request and retrieves the result from predefined action or default response.
        This method posts a chat request based on the context and messages provided and
        applies an action if it is defined. If no action is specified, it returns a default result.
        
        Args:
            context (LLMContext):
                 An instance of LLMContext that contains the context for the chat request.
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage instances to be sent in the chat request.
            **kwargs (Any):
                 Variable keyword arguments that can be used to provide additional settings or options.
        
        Returns:
            (LLMResult):
                 An instantiation of LLMResult containing the choices produced by the
                action or defaults to a message indicating the class name in case no
                action is defined.

        """
        if self._action is not None:
            return LLMResult(choices=self._action(messages))
        return LLMResult(choices=[f"{self.__class__.__name__}"])

    @staticmethod
    def from_responses(responses: List[str]) -> MockLLM:
        """
        Creates an instance of MockLLM with a predefined list of responses.
        
        Args:
            responses (List[str]):
                 A list of string responses to be used when mocking LLM interactions.
        
        Returns:
            (MockLLM):
                 An instance of the MockLLM class which uses the provided list of responses
                to generate the output of the '_post_chat_request' method.
        
        Note:
            The 'from_responses' method is intended to be used for testing purposes, providing a
            simple way to simulate LLM behavior with a set of predetermined responses.

        """
        return MockLLM(action=(lambda x: responses))

    @staticmethod
    def from_response(response: str) -> MockLLM:
        """
        Creates an instance of the MockLLM class with a predefined single-line response.
        This static method allows the instantiation of a MockLLM object that will return a specific
        response string for any given input messages. The MockLLM object created is capable of
        simulating a response from an LLM without actually sending a network request.
        
        Args:
            response (str):
                 A string representing the single-line response that the MockLLM
                instance should return when its `_post_chat_request` method is called.
        
        Returns:
            (MockLLM):
                 An instance of the MockLLM class with its action set to return
                the specified response string within a list.
            

        """
        return MockLLM(action=(lambda x: [response]))

    @staticmethod
    def from_multi_line_response(responses: Iterable[str]) -> MockLLM:
        """
        Creates a new `MockLLM` instance with a predefined multiline response.
        This static method takes an iterable of strings, joins them into a single string with newline characters separating each line,
        and then sets this as the response action for the `MockLLM` instance. When the `_post_chat_request` method is invoked,
        the instance will return a `LLMResult` with the multiline response as one of its choices.
        
        Parameters:
            responses (Iterable[str]):
                 An iterable of strings that will be combined into a single multiline response.
        
        Returns:
            (MockLLM):
                 An instance of `MockLLM` with a predefined action that generates the supplied multiline response.
            

        """
        response = "\n".join(responses)
        return MockLLM(action=(lambda x: [response]))


class MockErrorLLM(LLMBase):
    """
    A mock class that always raises a predefined exception when attempting to post a chat request.
    This class is intended for use in testing error handling by simulating an LLM (Language Learning Model) that
    always fails to post a chat request due to an exception. It extends the `LLMBase` class, allowing it to
    be used as a drop-in replacement for other LLM-based classes that normally send chat requests.
    
    Attributes:
        exception (LLMException):
             An instance of `LLMException` that is raised whenever `_post_chat_request`
            is called.
        
    
    Methods:
        __init__:
             Constructs the `MockErrorLLM` class with an optional exception parameter.
        _post_chat_request:
             Overrides the parent class method to always raise the predefined exception.
    
    Raises:
        LLMException:
             The predefined exception stored in the `exception` attribute when `_post_chat_request`
            is invoked.
        

    """
    def __init__(self, exception: LLMException = LLMException("From Mock", "mock")):
        """
        Initializes a new instance of a class that captures an LLMException.
        This method is a constructor for a class that is designed to encapsulate an exception
        specifically associated with LLM (Language Learning Models or similar context).
        It allows for optional customization of the exception message based on the LLM name.
        
        Args:
            exception (LLMException, optional):
                 The LLMException instance to be associated with
                this class instance. It defaults to a new instance
                of LLMException with a generic message 'From Mock'
                and an 'mock' llm name.
            

        """
        super().__init__()
        self.exception = exception

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a request to the chat model with the provided context and messages.
        This internal method prepares and sends a request to the language model chat API based on the specified context and sequence of messages. It accepts additional keyword arguments that may be used for request configuration. The method is designed to be overridden and should raise an exception if called directly, as it serves as a placeholder for actual implementation in subclasses.
        
        Args:
            self:
                 The instance of the class from which the method is called.
            context (LLMContext):
                 An instance of LLMContext which contains the context in which the chat is occurring.
            messages (Sequence[LLMMessage]):
                 A sequence of LLMMessage instances which represent the messages exchanged in the chat.
            **kwargs (Any):
                 Additional keyword arguments that may be used for fine-tuning the chat request.
        
        Returns:
            (LLMResult):
                 The result object returned from the chat model after processing the request.
        
        Raises:
            NotImplementedError:
                 If this method is called without being overridden in a subclass.
            

        """
        raise self.exception


class MockErrorSimilarityScorer(ScorerBase):
    """
    A mock scorer class that inherits from ScorerBase, meant to simulate error raising during the scoring process.
    This class is designed for testing purposes, where it raises a predefined exception
    when the `_score` method is called. The exception to be raised can be specified
    during instantiation. If no exception is specified, a generic Exception object is used by default.
    
    Attributes:
        exception (Exception):
             The exception instance that will be raised when `_score` is called.
    
    Args:
        exception (Exception, optional):
             An exception instance to be raised when `_score` is invoked.
            Defaults to Exception().
        

    """
    def __init__(self, exception: Exception = Exception()):
        """
        Initializes a new instance of the custom exception wrapper class.
        
        Args:
            exception (Exception, optional):
                 The exception object to be wrapped. Defaults to a new instance of Exception.

        """
        super().__init__()
        self.exception = exception

    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        Scores the given chat message based on the specified context.
        This method takes a ScorerContext object and a ChatMessage object, then
        computes and returns a floating-point score that represents a certain metric or
        quality of the chat message in that context. The exact nature of the metric is
        determined by the specific implementation of this method.
        The method is intended to be overridden by subclasses, and the default
        implementation simply raises an exception as it is meant to be abstract.
        
        Args:
            context (ScorerContext):
                 The context in which the scoring is occurring,
                which provides necessary information for score calculation.
            message (ChatMessage):
                 The message that is being scored.
        
        Returns:
            (float):
                 The calculated score representing the quality or metric of the chat
                message.
        
        Raises:
            self.exception:
                 An exception is raised if this method is not implemented
                in a subclass.

        """
        raise self.exception


class MockAgent(Agent):
    # noinspection PyMissingConstructor
    """
    A mock agent used for simulating an agent's behavior in a controlled environment.
    The `MockAgent` class inherits from the base `Agent` class and is designed to mimic an agent's actions. Its primary purpose is for testing and development.
    When an instance of `MockAgent` executes, it sleeps for a random amount of time within a specified range before returning a predefined message wrapped in an `AgentResult`.
    
    Attributes:
        message (str):
             A default message that the mock agent will return during execution. Defaults to 'agent message'.
        data (Any):
             Additional data that can be associated with the message. Defaults to None.
        score (float):
             A score value associated with the message. Defaults to 1.0.
        sleep (float):
             The minimum amount of time in seconds that the agent will sleep before executing. Defaults to 0.2.
        sleep_interval (float):
             The additional random amount of time in seconds that will be added to `sleep` for the agent to sleep. Defaults to 0.1.
    
    Methods:
        __init__:
             Constructs a new instance of MockAgent with provided message, data, score, and sleep parameters.
        execute:
             Simulates the agent's execution within the given context and budget, sleeping for a random time before returning the result.
        

    """
    def __init__(
        self,
        message: str = "agent message",
        data: Any = None,
        score: float = 1.0,
        sleep: float = 0.2,
        sleep_interval: float = 0.1,
    ):
        """
        Initializes the class with specified parameters.
        
        Args:
            message (str, optional):
                 A string message representing the agent message. Default is 'agent message'.
            data (Any, optional):
                 Data payload relevant to the instance being created. It can be of any type. Default is None.
            score (float, optional):
                 A float score value related to the instance. Default is 1.0.
            sleep (float, optional):
                 A float value indicating the duration the agent should sleep/wait. Default is 0.2 seconds.
            sleep_interval (float, optional):
                 A float value that specifies the interval between sleep checks. Default is 0.1 seconds.

        """
        self.message = message
        self.data = data
        self.score = score
        self.sleep = sleep
        self.sleep_interval = sleep_interval

    def execute(self, context: AgentContext, budget: Optional[Budget] = None) -> AgentResult:
        """
        Executes the agent with a given context and an optional budget. The function triggers a random sleep
        period based on agent's settings before creating and returning an AgentResult.
        This method simulates execution time by sleeping for a random duration computed from the
        agent's sleep settings (self.sleep to self.sleep + self.sleep_interval). After the sleep period,
        it constructs a ScoredChatMessage using the agent's preset message and data, assigning it the
        pre-configured score. This message is then encapsulated in an AgentResult object which is returned.
        
        Args:
            context (AgentContext):
                 The context in which the agent operates, providing relevant information
                that influences agent execution.
            budget (Optional[Budget]):
                 An optional budget for the execution, may impose resource constraints.
        
        Returns:
            (AgentResult):
                 The result of the agent's execution, containing messages scored based on predefined criteria.
            

        """
        time.sleep(random.uniform(self.sleep, self.sleep + self.sleep_interval))
        return AgentResult([ScoredChatMessage(ChatMessage.agent(self.message, self.data), score=self.score)])


class MockErrorAgent(Agent):
    # noinspection PyMissingConstructor
    """
    A class that represents a mock agent used for simulating error conditions during agent execution.
    This agent is intended for testing scenarios where an error or exception needs to be raised instead of completing the execution successfully. When `execute` method is called on an instance of this class, it will raise the specified exception.
    
    Attributes:
        exception (Exception):
             The exception that will be raised when the `execute` method is called. Defaults to a base Exception instance.
    
    Args:
        exception (Exception, optional):
             The specific exception to use when the `execute` method is invoked. Defaults to Exception().
        

    """
    def __init__(self, exception: Exception = Exception()):
        """
        Initializes a new instance of the FilterException class.
        
        Args:
            exception (Exception, optional):
                 The exception object that caused the filter exception.
                Defaults to an instance of the base Exception class.

        """
        self.exception = exception

    def execute(self, context: AgentContext, budget: Optional[Budget] = None) -> AgentResult:
        """
        Executes an action in a given context with an optional budget constraint.

        """
        raise self.exception


class MockMonitored(Monitored):
    """
    A subclass of Monitored that creates a mock object for monitoring purposes.
    This class is specifically designed for testing and development scenarios where a placeholder object with monitoring capabilities is required. It inherits from the Monitored class and passes a default name and a Monitorable object with a mock identifier to the superclass constructor.
    
    Attributes:
        Inherits all attributes from the Monitored class.
    
    Args:
        name (str, optional):
             The name of the mock monitored object. Defaults to 'mock'.

    """
    def __init__(self, name: str = "mock"):
        """
        Initialize a new instance of the class with the given name.
        This constructor initializes the object by assigning the provided name to it,
        and setting up monitoring with a Monitorable instance using 'mock' as its identifier.
        
        Args:
            name (str):
                 The name to assign to this instance. Defaults to 'mock'.
            

        """
        super().__init__(name, Monitorable("mock"))
