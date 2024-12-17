from __future__ import annotations

import os
from typing import Any, Generic, Optional, Protocol, Sequence, Type, TypeVar, Union, cast

from council import LLMContext
from council.llm import (
    LLMBase,
    LLMFunction,
    LLMMessage,
    LLMMiddlewareChain,
    LLMParsingException,
    LLMResponse,
    get_llm_from_config,
)
from typing_extensions import Self


class ChainableLLMFunctionInput(Protocol):
    """Input for a chainable LLM function."""

    def to_prompt(self) -> str:
        """Convert the object to a prompt string."""
        ...


class ChainableLLMFunctionOutput(Protocol):
    """Output of a chainable LLM function."""

    @classmethod
    def from_response(cls, response: LLMResponse) -> Self:
        """Parse the object from an LLM response."""
        ...

    @classmethod
    def to_response_template(cls) -> str:
        """Convert the object to a response template."""
        ...


T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")

T_LLMInput = TypeVar("T_LLMInput", bound=ChainableLLMFunctionInput)
T_LLMOutput = TypeVar("T_LLMOutput", bound=ChainableLLMFunctionOutput)


class ChainableFunction(Generic[T_Input, T_Output]):
    def execute(self, obj: T_Input, exception: Optional[Exception] = None) -> T_Output:
        # Here you'd implement generic logic that transforms obj into T_Output
        raise NotImplementedError()


class ChainableLLMFunction(ChainableFunction[T_LLMInput, T_LLMOutput]):
    PROMPT_TEMPLATE = "\n".join(
        ["{input_obj_prompt}", "", "{response_template}"]
    )  # TODO: could extend with {additional_instructions}

    def __init__(
        self, llm: Union[LLMBase, LLMMiddlewareChain], output_obj_type: Type[T_LLMOutput], max_retries: int = 3
    ):
        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._output_obj_type = output_obj_type
        self._context = LLMContext.empty()

        self._max_retries = max_retries

    @classmethod
    def from_config(
        cls,
        output_obj_type: Type[T_LLMOutput],
        path_prefix: str = "data",
        llm_config_path: str = "llm-config.yaml",
        max_retries: int = 3,
    ) -> ChainableLLMFunction:
        llm = get_llm_from_config(os.path.join(path_prefix, llm_config_path))
        return ChainableLLMFunction(llm, output_obj_type, max_retries)

    def execute(self, obj: T_LLMInput, exception: Optional[Exception] = None) -> T_LLMOutput:
        input_prompt = self.PROMPT_TEMPLATE.format(
            input_obj_prompt=obj.to_prompt(), response_template=self._output_obj_type.to_response_template()
        )
        messages = [LLMMessage.user_message(input_prompt)]

        llm_func: LLMFunction[T_LLMOutput] = LLMFunction(
            llm=self._llm_middleware,
            response_parser=self._output_obj_type.from_response,
            messages=messages,
            max_retries=self._max_retries,
        )
        return llm_func.execute("")


class FunctionChainBase(Generic[T_Input, T_Output]):
    def __init__(self, functions: Sequence[ChainableFunction]):
        self.functions = functions

    def execute(self, obj: T_Input) -> T_Output:
        raise NotImplementedError()


class LinearFunctionChain(FunctionChainBase[T_Input, T_Output]):
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
        current_obj: Any = obj
        for func in self.functions:
            current_obj = func.execute(current_obj)
        return cast(T_Output, current_obj)


class BacktrackFunctionChain(FunctionChainBase[T_Input, T_Output]):
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
        results = []
        current_obj: Any = obj
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

        return cast(T_Output, results[-1])


# TODO: # an example for sql generation and execution where limit must be applied
