"""


Module `llm_base`

This module defines the abstract base for handling requests and responses within a chat-based interface specifically designed for processing and interacting with messages.

Classes:
    LLMResult: Encapsulates the result of a request containing possible choices and any associated consumptions.
    LLMBase: Abstract base class that provides a common interface for monitoring and handling the life cycle of a chat request.

Imported Entities:
    abc: Abstract Base Classes module, used for defining abstract methods in classes.
    Any, Optional, Sequence: Typing modules, used for indicating type hints.
    Consumption, LLMContext, Monitorable: Custom classes imported from other modules or packages to be used for defining consumptions and providing monitoring interfaces.
    LLMessageTokenCounterBase, LLMMessage: Custom classes used to count tokens in messages and encapsulate message information respectively.

Notable Attributes and Methods:
    - LLMBase._post_chat_request: An abstract method that must be implemented by subclasses, defining how chat requests are processed.
    - LLMBase.post_chat_request: Method that includes pre-processing like token counting and wraps the call to the abstract `_post_chat_request` method within a context, logging execution details.


"""
import abc
from typing import Any, Optional, Sequence

from council.contexts import Consumption, LLMContext, Monitorable

from .llm_message import LLMessageTokenCounterBase, LLMMessage


class LLMResult:
    """
    Class to represent the result from a Large Language Model (LLM). It contains the choices provided by the LLM as well as the associated resource consumptions if available.
    
    Attributes:
        _choices (List[str]):
             The list of choices provided by the LLM. This list should not be empty.
        _consumptions (List[Consumption], optional):
             The list of resource consumption objects, which may include metrics such as processing time,
            memory usage, or other relevant consumption data.
    
    Methods:
        __init__(choices:
             Sequence[str], consumptions: Optional[Sequence[Consumption]] = None):
            Initializes a new instance of the LLMResult class.
    
    Args:
        choices (Sequence[str]):
             A sequence of strings representing the choices returned by the LLM.
        consumptions (Optional[Sequence[Consumption]]):
             An optional sequence of Consumption objects representing the resources
            consumed while generating the choices. Defaults to None.
        first_choice (str):
             Returns the first choice from the list of choices. This is equivalent to accessing the first element of
            the _choices attribute.
    
    Returns:
        (str):
             The first choice provided by the LLM.
        choices (Sequence[str]):
             Provides access to the _choices attribute.
    
    Returns:
        (Sequence[str]):
             The sequence of choices returned by the LLM.
        consumptions (Sequence[Consumption]):
             Provides access to the _consumptions attribute.
    
    Returns:
        (Sequence[Consumption]):
             The sequence of Consumption objects, or an empty list if no consumptions data were provided.

    """
    def __init__(self, choices: Sequence[str], consumptions: Optional[Sequence[Consumption]] = None):
        """
        Initializes a new instance of the class with specific choices and consumption options.
        
        Args:
            choices (Sequence[str]):
                 A list of strings representing the available choices.
            consumptions (Optional[Sequence[Consumption]]):
                 A list of Consumption objects
                representing the resource consumptions associated with each choice. If
                `None`, the consumptions list will be empty.

        """
        self._choices = list(choices)
        self._consumptions = list(consumptions) if consumptions is not None else []

    @property
    def first_choice(self) -> str:
        """
        
        Returns the first choice from a list of choices stored within the property _choices.
        
        Returns:
            (str):
                 The first element from the _choices list.

        """
        return self._choices[0]

    @property
    def choices(self) -> Sequence[str]:
        """
        
        Returns the sequence of choices available in the object. The choices that can be accessed are stored in a private variable and made available as a read-only property through this method. This can be used to expose the available options without allowing direct modification of the underlying data structure that's holding these choices.
        
        Returns:
            (Sequence[str]):
                 A sequence (e.g., a list or a tuple) containing the choice strings.

        """
        return self._choices

    @property
    def consumptions(self) -> Sequence[Consumption]:
        """
        
        Returns the sequence of Consumption objects associated with the current instance.
            This property provides access to an internal sequence that contains Consumption objects,
            providing information about consumptions tracked by this instance.
        
        Returns:
            (Sequence[Consumption]):
                 A sequence of Consumption objects.

        """
        return self._consumptions


class LLMBase(Monitorable, abc.ABC):
    """
    class LLMBase(Monitorable, abc.ABC):
    
    Represents the base class for a language model's lifecycle management.
    This class is responsible for handling the high-level logic of processing chat
    requests and tracking token usage if a token counter is provided. It ensures
    that consumptions are recorded as part of the context budget and properly logs
    the request execution flow. This class needs to be extended with an implementation
    of the `_post_chat_request` method for the specific language model behavior.
    
    Attributes:
        _token_counter (Optional[LLMessageTokenCounterBase]):
             A counter to track the number of tokens used.
            Defaults to None if not provided.
        _name (str):
             The name identifier for the language model instance, automatically
            generated based on the class name if not explicitly supplied.
    
    Args:
        token_counter (Optional[LLMessageTokenCounterBase]):
             The token counter instance to be
            used by the language model, if required. Defaults to None.
        name (Optional[str]):
             Optional name for the language model instance. If not provided, a default
            name is constructed from the class name.
    
    Methods:
        post_chat_request:
            Processes a chat request within the given context and with the provided messages.
            It tracks the token consumption, logs the request processing, and ensures
            the context budget is updated accordingly. If a `_token_counter` is set,
            it uses it to count the tokens in the messages provided.
    
    Args:
        context (LLMContext):
             The context in which the chat request is executed.
        messages (Sequence[LLMMessage]):
             A sequence of messages that are part of the chat request.
        **kwargs (Any):
             Additional keyword arguments that can be used by the `_post_chat_request` method.
    
    Returns:
        (LLMResult):
             The result of the language model processing the request.
    
    Raises:
        Exception:
             If any errors occur during the processing of the chat request.
        _post_chat_request:
            Abstract method to be implemented by the subclass that actually handles the particulars
            of the chat request. This method is called by `post_chat_request`.
    
    Args:
        context (LLMContext):
             The context in which the chat request is executed.
        messages (Sequence[LLMMessage]):
             A sequence of messages that are part of the chat request.
        **kwargs (Any):
             Additional keyword arguments specific to the subclass implementation.
    
    Returns:
        (LLMResult):
             The outcome of the chat request as processed by the language model.
    
    Raises:
        NotImplementedError:
             If the subclass does not provide an implementation.
        

    """

    def __init__(self, token_counter: Optional[LLMessageTokenCounterBase] = None, name: Optional[str] = None):
        """
        @property
        The default_execution_unit_rank is a property that allows you to get the execution rank of the unit.
        This method checks if parallelism is enabled (indicated by the presence of the
        _parallelism attribute) and returns the default rank for execution units accordingly.
        
        Returns:
            (Optional[int]):
                 Returns 1 if parallelism is enabled, otherwise returns None.

        """
        super().__init__(name or "llm")
        self._token_counter = token_counter
        self._name = name or f"llm_{self.__class__.__name__}"

    def post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request and handles token counting, logging, and execution within a context.
        This method initiates the chat request process by first counting the number of tokens in the
        messages using the `_token_counter` if available. It then logs the start of the request
        execution, tries to post the chat request by calling `_post_chat_request`, updates the budget
        with consumption details, and finally returns the result. If the request execution fails for
        any reason, it logs the exception and re-raises it. The method also ensures that a final log
        entry is made once the execution is completed, regardless of its success or failure.
        
        Args:
            context (LLMContext):
                 The context in which the chat request is executed.
                It provides logging capabilities and can manage resources within its scope.
            messages (Sequence[LLMMessage]):
                 A sequence of message objects that will be sent in the
                chat request.
            **kwargs (Any):
                 Additional keyword arguments that will be passed directly to the
                `_post_chat_request` method.
        
        Raises:
            Exception:
                 If the request execution fails, the exception is logged and re-raised.
        
        Returns:
            (LLMResult):
                 The result of the chat request, including any consumptions.
            

        """

        if self._token_counter is not None:
            _ = self._token_counter.count_messages_token(messages=messages)

        context.logger.debug(f'message="starting execution of llm {self._name} request"')
        try:
            with context:
                result = self._post_chat_request(context, messages, **kwargs)
                context.budget.add_consumptions(result.consumptions)
                return result
        except Exception as e:
            context.logger.exception(f'message="failed execution of llm {self._name} request" exception="{e}" ')
            raise e
        finally:
            context.logger.debug(f'message="done execution of llm {self._name} request"')

    @abc.abstractmethod
    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the underlying service specific to the language model implementation.
        This abstract method must be implemented by subclasses to initiate a chat request with the given context and sequence of messages. The method can accept additional keyword arguments suitable for the implementation details of the service being interacted with.
        
        Args:
            context (LLMContext):
                 The context object containing information specific to the language model's environment and state.
            messages (Sequence[LLMMessage]):
                 A sequence of message objects that encapsulate the dialogue or conversation intended to be sent to the language model.
            **kwargs (Any):
                 Variable length keyword arguments that are specific to the language model implementation and the underlying service requirements.
        
        Returns:
            (LLMResult):
                 The result object which encapsulates the response from the language model after processing the chat request.
        
        Raises:
            NotImplementedError:
                 If the method is not implemented by a subclass.

        """
        pass
