from typing import Protocol, Generic, TypeVar, Optional, Sequence, Any

from typing_extensions import Self

from council.llm import LLMResponse


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
    def execute(self, obj: T_LLMInput, exception: Optional[Exception] = None) -> T_LLMOutput:
        # Here you'd implement logic that uses obj.to_prompt(),
        # calls some LLM, and returns a T_LLMOutput object that can .to_response()
        raise NotImplementedError()


class FunctionChain(Generic[T_Input, T_Output]):
    def __init__(self, functions: Sequence[ChainableFunction]):
        self.functions = functions

    def execute(self, obj: T_Input) -> T_Output:
        current = obj
        for func in self.functions:
            current = func.execute(current)
        return current  # TODO: # type: ignore
