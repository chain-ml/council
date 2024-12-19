from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar, Union, cast

from council.llm import LLMBase, LLMFunction, LLMMessage, LLMMiddlewareChain
from council.llm.llm_response_parser import BaseModelResponseParser


class LinkProcessorException(Exception):
    """
    Exception raised during the execution of a LinkProcessor.
    Contains information about the input that caused the exception, the exception itself,
    and optionally a previous processor that should handle the exception.
    """

    def __init__(self, *, input: str, message: str, transfer_to: Optional[str] = None):
        self.input = input
        self.message = message
        # self.transfer_to = transfer_to


class LinkLLMProcessorInput(Protocol):
    """Input for a LinkLLMProcessor."""

    def to_prompt(self) -> str:
        """Convert the object to a prompt string."""
        ...


T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")

T_LLMInput = TypeVar("T_LLMInput", bound=LinkLLMProcessorInput)
T_LLMOutput = TypeVar("T_LLMOutput", bound=BaseModelResponseParser)


class LinkProcessor(Generic[T_Input, T_Output]):
    """
    Base class for a LinkProcessor, converting an input object into an output object.
    """

    def execute(self, obj: T_Input, exception: Optional[LinkProcessorException] = None) -> T_Output:
        # implement logic that transforms obj into T_Output
        raise NotImplementedError()


class LinkLLMProcessor(LinkProcessor[T_LLMInput, T_LLMOutput]):
    """
    LinkProcessor that uses an LLM to convert an input object into an output object.
    Keeps track of exceptions occurring for this instance.
    """

    PROMPT_TEMPLATE = "\n".join(["{input_obj_prompt}", "", "{response_template}"])

    def __init__(self, llm: Union[LLMBase, LLMMiddlewareChain], output_obj_type: Type[T_LLMOutput]):
        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._output_obj_type = output_obj_type

        # store all input-output pairs instead of only exception ones?
        self._memory: List[Dict[str, str]] = []
        self._previous_exceptions: List[LinkProcessorException] = []

    @property
    def memory(self) -> List[Dict[str, str]]:
        return self._memory

    @property
    def exceptions_memory(self) -> List[Dict[str, str]]:
        return [memory for memory in self._memory if "exception" in memory]

    def add_to_memory(self, input_obj: T_LLMInput, output: Any) -> None:
        self._memory.append(
            {
                "input": input_obj.to_prompt(),
                "output": output,
            }
        )

    @property
    def previous_exceptions(self) -> List[LinkProcessorException]:
        return self._previous_exceptions

    def add_previous_exception(self, exception: LinkProcessorException):
        self._previous_exceptions.append(exception)

    def execute(self, obj: T_LLMInput, exception: Optional[LinkProcessorException] = None) -> T_LLMOutput:
        system_prompt = self.PROMPT_TEMPLATE.format(
            input_obj_prompt=obj.to_prompt(),
            response_template=self._output_obj_type.to_response_template(),
        )
        messages = [LLMMessage.system_message(system_prompt)]
        if exception is not None:
            self.add_previous_exception(exception)
            messages.extend([LLMMessage.assistant_message(exception.input), LLMMessage.user_message(exception.message)])

        llm_func: LLMFunction[T_LLMOutput] = LLMFunction(
            llm=self._llm_middleware,
            response_parser=self._output_obj_type.from_response,
            messages=messages,
        )
        return llm_func.execute("")  # TODO:


class ChainProcessorBase(Generic[T_Input, T_Output]):
    """
    Base class for a ChainProcessor, executing a list of LinkProcessors in sequence.
    """

    def __init__(self, processors: List[LinkProcessor]):
        self.processors = processors

    def execute(self, obj: T_Input) -> T_Output:
        raise NotImplementedError()


class LinearChainProcessor(ChainProcessorBase[T_Input, T_Output]):
    """
    ChainProcessor that executes links in a linear order without any error handling.
    Each link should be able to handle errors independently.

    .. mermaid::

        flowchart LR
            A(Link A) --> B(Link B)
            A --> A
            B --> B
    """

    def execute(self, obj: T_Input) -> T_Output:
        current_obj: Any = obj
        for processor in self.processors:
            current_obj = processor.execute(current_obj)
        return cast(T_Output, current_obj)


class BacktrackingChainProcessor(ChainProcessorBase[T_Input, T_Output]):
    """
    ChainProcessor that executes links in a linear order backtracking errors.
    If a link fails, the chain will backtrack to the previous link and try again.

    .. mermaid::

        flowchart LR
            A(Link A) --> B(Link B)
            A --> A
            B --> B
            B --> A
    """

    def __init__(self, processors: List[LinkProcessor], max_backtracks: int = 3):
        super().__init__(processors)
        self.max_backtracks = max_backtracks

    def execute(self, obj: T_Input) -> T_Output:
        index = 0
        inputs: List[T_Input] = [obj] * len(self.processors)
        previous_exception: Optional[LinkProcessorException] = None
        current_obj = obj
        backtrack_count = 0

        while index < len(self.processors):
            try:
                current_obj = self.processors[index].execute(inputs[index], previous_exception)
                index += 1
                if index < len(self.processors):
                    inputs[index] = current_obj
            except LinkProcessorException as e:
                if backtrack_count >= self.max_backtracks:
                    raise e  # out of retry

                index -= 1
                backtrack_count += 1
                previous_exception = e

        return cast(T_Output, current_obj)
