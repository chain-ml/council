"""

The `openai_chat_completions_llm` module provides an interface for interacting with the OpenAI Chat Completions API using HTTP requests. It includes classes to construct requests, parse responses, and encapsulate the results with proper handling of token counts and consumptions which is important for managing API usage and cost. The module defines a Protocol for providers and supports extension for various configuration setups through `LLMBase` implementations.

Classes:
    Provider -- A Protocol that specifies a callable interface for making HTTP requests.

    Message -- Represents a single chat message with a role (e.g., user or system) and content.

    Choice -- Represents a choice from the completion response, including a message and metadata like the finish reason and index.

    Usage -- Encapsulates the token usage of the completion including the number of tokens used by the prompt, completion, and their total.

    OpenAIChatCompletionsResult -- Encapsulates the result from the OpenAI chat completions call including identification, model information, choices, and usage statistics.

    OpenAIChatCompletionsModel -- Inherits from LLMBase and is responsible for constructing the payload for the chat completion request, making the HTTP request, handling the response, and converting it into an LLMResult.

The module also imports necessary components from other modules such as `httpx`, `LLMConfigurationBase`, `LLMMessage`, `LLMessageTokenCounterBase`, `LLMException`, `LLMBase`, and `LLMResult`, as well as context and consumption classes from the `council.contexts` module.


"""
from __future__ import annotations

import httpx

from typing import List, Any, Protocol, Sequence, Optional

from . import LLMConfigurationBase
from .llm_message import LLMMessage, LLMessageTokenCounterBase
from .llm_exception import LLMCallException
from .llm_base import LLMBase, LLMResult
from council.contexts import LLMContext, Consumption


class Provider(Protocol):
    """
    A protocol class that represents a callable provider which takes a payload and returns an HTTP response.
    This class is meant to be used as a contract for implementation by other classes that will act as a provider for HTTP-based services. The __call__ method must be implemented by any class that conforms to this protocol, enabling the class to be called with a payload and in turn, provide an HTTP response.

    Attributes:
        Not applicable for protocol classes.

    Methods:
        __call__(payload:
             dict[str, Any]) -> httpx.Response:
            When implemented, this method will take a payload as a dictionary where keys are strings and values are of any type, and it should return an httpx.Response object. This allows interaction with HTTP services in a standardized way.

    Args:
        payload (dict[str, Any]):
             A dictionary containing the data to be sent to an HTTP service.

    Returns:
        (httpx.Response):
             The response from the HTTP service.

    """

    def __call__(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Calls the instance like a function, sending the given payload as a request to a predefined endpoint.
        This special method allows the object to be called with parentheses and pass a payload to send a
        request. It is commonly used in classes that represent services or API clients. The method should
        handle request sending logic internally and return the response received from the service. This is
        useful when the class instance is meant to encapsulate the behavior of a callable object.

        Args:
            payload (dict[str, Any]):
                 The data to be sent in the request. This dictionary should conform
                to the expectations of the endpoint being called (e.g., required fields, value types).

        Returns:
            (httpx.Response):
                 The response object received from the endpoint after sending the request.
                This object will contain the HTTP status code, response headers, and body content.

        """
        ...


class Message:
    """
    A class that encapsulates a message with a specified role and content.
    
    Attributes:
        _role (str):
             The role associated with the message indicating the context or the sender.
        _content (str):
             The actual textual content of the message.
    
    Methods:
        __init__:
             Constructor that initializes a Message instance with role and content.
        content (property):
             A property that gets the message's content.
        from_dict:
             A static method that creates a Message instance from a dictionary object.

    """
    _role: str
    _content: str

    def __init__(self, role: str, content: str):
        """
        Initializes an instance of the class with specified role and content attributes.
        
        Args:
            role (str):
                 A string representing the role of the object or the user.
            content (str):
                 A string containing the information or content related to the role.
                This constructor method sets up the object with the given role and content parameters,
                allowing for the object's state to be initialized properly.

        """
        self._content = content
        self._role = role

    @property
    def content(self) -> str:
        """
        Property that gets the private `_content` attribute.
        This property is used to retrieve the value of the private `_content` attribute
        which represents some form of content encapsulated within the class.
        
        Returns:
            (str):
                 The current value of the `_content` attribute.
            

        """
        return self._content

    @staticmethod
    def from_dict(obj: Any) -> Message:
        """
        Converts a dictionary to a `Message` instance.
        This static method takes a dictionary with keys 'role' and 'content',
        and constructs a `Message` object using these values. The dictionary
        values corresponding to the keys 'role' and 'content' are converted to
        strings before instantiation of the `Message` object.
        
        Args:
            obj (Any):
                 The dictionary object to be converted, which should
                contain the 'role' and 'content' keys.
        
        Returns:
            (Message):
                 An instance of the `Message` class populated with the
                'role' and 'content' values from the input dictionary.

        """
        _role = str(obj.get("role"))
        _content = str(obj.get("content"))
        return Message(_role, _content)


class Choice:
    """
    A class representing a choice with a message, provided index and reason for finishing.
    
    Attributes:
        _index (int):
             An integer representing the index of the choice.
        _finish_reason (str):
             A string describing the reason why the choice is a finishing move.
        _message (Message):
             A Message object associated with the choice containing further details.
    
    Methods:
        __init__:
             Constructs a new Choice instance.
        message:
             A property that returns the associated Message object.
        from_dict:
             A static method that constructs a Choice instance from a dictionary object.

    """
    _index: int
    _finish_reason: str
    _message: Message

    def __init__(self, index: int, finish_reason: str, message: Message):
        """
        Initializes the instance with an index, finish reason, and a message object.
        
        Args:
            index (int):
                 An integer that represents the index or unique identifier.
            finish_reason (str):
                 A string that describes the reason why a process finished.
            message (Message):
                 An object of type Message that contains the message details.
        
        Attributes:
            _index (int):
                 Internal storage of the index passed during initialization.
            _finish_reason (str):
                 Internal storage of the finish reason.
            _message (Message):
                 Internal storage of the message object.
            

        """
        self._index = index
        self._finish_reason = finish_reason
        self._message = message

    @property
    def message(self) -> Message:
        """
        Gets the current message held in this instance.
        This is a property method that returns the message object associated with an instance.
        It is accessed like an attribute, but it is actually a method that when called, will return the message object.
        
        Note that this is a read-only property, so it cannot be set directly.
        
        Returns:
            (Message):
                 The message object associated with this instance.

        """
        return self._message

    @staticmethod
    def from_dict(obj: Any) -> Choice:
        """
        Converts a dictionary representation of a Choice into a Choice object.
        This static method takes a dictionary with keys that represent the attributes of a Choice
        object and constructs a Choice instance with those attributes.
        
        Args:
            obj (Any):
                 A dictionary with keys 'index', 'finish_reason', and 'message' that correspond to
                the attributes of Choice. The 'index' key should map to an integer, 'finish_reason' to
                a string, and 'message' to a dictionary that can be converted to a Message instance
                using Message.from_dict.
        
        Returns:
            (Choice):
                 A Choice object initialized with the attributes provided in the input dictionary.
        
        Raises:
            ValueError:
                 If the 'index' key is not present or not an integer.
            TypeError:
                 If the 'finish_reason' key is not present or not a string.
            KeyError:
                 If the 'message' key is not present or its value cannot be converted to a
                Message instance by Message.from_dict.
            

        """
        _index = int(obj.get("index"))
        _finish_reason = str(obj.get("finish_reason"))
        _message = Message.from_dict(obj.get("message"))
        return Choice(_index, _finish_reason, _message)


class Usage:
    """
    A class representing the Usage data structure, which contains information about token usage counts in requests.
    This class provides a structured way to access and manage token usage data. It operates on three
    integers representing the number of completion tokens, prompt tokens, and the total tokens used in a
    request. The class includes properties to retrieve these values and a static method to create a Usage
    instance from a dictionary object.
    
    Attributes:
        _completion (int):
             Number of tokens used for completions.
        _prompt (int):
             Number of tokens used for prompting.
        _total (int):
             Total number of tokens used.
    
    Methods:
        __init__(completion_tokens:
             int, prompt_tokens: int, total_tokens: int):
            Initializes a new instance of Usage.
        __str__() -> str:
            Provides a string representation of the Usage instance.
        prompt_tokens() -> int:
            Property to get the number of prompt tokens used.
        completion_tokens() -> int:
            Property to get the number of completion tokens used.
        total_tokens() -> int:
            Property to get the total number of tokens used.
        from_dict(obj:
             Any) -> Usage:
            Static method that creates a Usage instance from a dictionary.

    """
    _completion: int
    _prompt: int
    _total: int

    def __init__(self, completion_tokens: int, prompt_tokens: int, total_tokens: int):
        """
        Initializes a new instance of the class with specified token counts.
        
        Args:
            completion_tokens (int):
                 The number of tokens generated in the completion.
            prompt_tokens (int):
                 The number of tokens used in the prompt.
            total_tokens (int):
                 The total number of tokens the model can handle.
        
        Attributes:
            _completion (int):
                 Internal storage for the count of completion tokens.
            _prompt (int):
                 Internal storage for the count of prompt tokens.
            _total (int):
                 Internal storage for the total token count.

        """
        self._completion = completion_tokens
        self._prompt = prompt_tokens
        self._total = total_tokens

    def __str__(self) -> str:
        """
        Converts the object to its string representation.
        This special method is used to create a human-readable string representation of the object whenever it is converted to a string, e.g., by the str() function or when it is printed. The string representation includes the prompt tokens, total tokens, and completion tokens of the object.
        
        Returns:
            (str):
                 The string representation of the object, including the prompt tokens, total tokens, and completion tokens.

        """
        return f'prompt_tokens="{self._prompt}" total_tokens="{self._total}" completion_tokens="{self._completion}"'

    @property
    def prompt_tokens(self) -> int:
        """
        Gets the number of prompt tokens. This property represents the number of tokens used in the prompt.
        
        Returns:
            (int):
                 The number of prompt tokens.

        """
        return self._prompt

    @property
    def completion_tokens(self) -> int:
        """
        Gets the number of completion tokens.
        This method is a property that, when accessed, will return the
        number of tokens that have been completed.
        
        Returns:
            (int):
                 The number of completion tokens.

        """
        return self._completion

    @property
    def total_tokens(self) -> int:
        """
        
        Returns the total number of tokens.
            This method is a property that represents the total number of tokens. It allows
            read-only access to the '_total' attribute which signifies the total count of
            tokens.
        
        Returns:
            (int):
                 The total number of tokens.

        """
        return self._total

    @staticmethod
    def from_dict(obj: Any) -> Usage:
        """
        Converts a dictionary representation of a Usage object into an actual Usage object instance.
        
        Args:
            obj (Any):
                 A dictionary that should have the keys 'completion_tokens',
                'prompt_tokens', and 'total_tokens' with their associated integer values.
        
        Returns:
            (Usage):
                 An instance of the Usage class initialized with the values extracted
                from the 'obj' dictionary.
        
        Raises:
            ValueError:
                 If any of the required keys ('completion_tokens', 'prompt_tokens',
                'total_tokens') are missing from 'obj' or they are not castable to an integer.

        """
        _completion_tokens = int(obj.get("completion_tokens"))
        _prompt_tokens = int(obj.get("prompt_tokens"))
        _total_tokens = int(obj.get("total_tokens"))
        return Usage(_completion_tokens, _prompt_tokens, _total_tokens)


class OpenAIChatCompletionsResult:
    """
    class OpenAIChatCompletionsResult:
    
    Represents the result data from an OpenAI chat completion request.
    
    Attributes:
        _id (str):
             A unique identifier for the chat completion result.
        _object (str):
             The object type, typically indicating it's a response or completion object.
        _created (int):
             A Unix timestamp indicating the creation time of the completion.
        _model (str):
             The model used for generating the chat completion.
        _choices (List[Choice]):
             A list of Choice objects, representing possible completion suggestions or messages.
        _usage (Usage):
             A summary of the API usage information for the current completion request.
    
    Methods:
        __init__(self, id:
             str, object: str, created: int, model: str, choices: List[Choice], usage: Usage):
            Initialize the OpenAIChatCompletionsResult instance with provided details.
        id(self) -> str:
            Property that returns the unique identifier of the chat completion result.
        model(self) -> str:
            Property that returns the model used for the chat completion.
        usage(self) -> Usage:
            Property that returns the API usage information of the completion.
        choices(self) -> Sequence[Choice]:
            Property that returns the list of Choice objects representing completion suggestions.
        to_consumptions(self) -> Sequence[Consumption]:
            Convert current completion usage data into a sequence of Consumption instances.
        from_dict(obj:
             Any) -> 'OpenAIChatCompletionsResult':
            Create an instance of OpenAIChatCompletionsResult from a dictionary object.

    """
    _id: str
    _object: str
    _created: int
    _model: str
    _choices: List[Choice]
    _usage: Usage

    def __init__(self, id: str, object: str, created: int, model: str, choices: List[Choice], usage: Usage):
        """
        Initializes a new instance of the class with various attributes related to the object.
        
        Args:
            id (str):
                 The unique identifier for the instance.
            object (str):
                 A string representing the type of object.
            created (int):
                 A Unix timestamp indicating when the object was created.
            model (str):
                 A string representing the model associated with the object.
            choices (List[Choice]):
                 A list of choices or options associated with the object.
            usage (Usage):
                 An instance of Usage class, describing how the object is used.
        
        Attributes:
            _id (str):
                 Stores the unique identifier for the instance.
            _object (str):
                 Stores the type of the object.
            _created (int):
                 Stores the Unix timestamp of when the object was created.
            _model (str):
                 Stores the model associated with the object.
            _choices (List[Choice]):
                 Stores the choices or options for the object.
            _usage (Usage):
                 Stores the usage information for the object.

        """
        self._id = id
        self._object = object
        self._usage = usage
        self._model = model
        self._choices = choices
        self._created = created

    @property
    def id(self) -> str:
        """
        Gets the unique identifier of the object.
        This property method returns the id attribute of the object, which is intended to be a unique identifier typically assigned upon object creation.
        
        Returns:
            (str):
                 The unique identifier of the object.

        """
        return self._id

    @property
    def model(self) -> str:
        """
        Property that gets the model name of an object.
        This read-only property retrieves the model name or designation that is internally
        stored in the private attribute '_model'. Typically, this property is used
        to expose the model information of an instance without granting direct
        access to the underlying data attribute, thus maintaining encapsulation.
        
        Returns:
            (str):
                 The model name or designation of the object.
            

        """
        return self._model

    @property
    def usage(self) -> Usage:
        """
        Property that gets the current usage status.
        This property allows the retrieval of the current usage state or statistics
        associated with an instance. It returns a 'Usage' object that contains
        the detailed usage information.
        
        Returns:
            (Usage):
                 An object encapsulating the usage data.
            

        """
        return self._usage

    @property
    def choices(self) -> Sequence[Choice]:
        """
        
        Returns the sequence of Choice objects available for the current instance.
            This property method provides read-only access to the sequence of choices.
        
        Returns:
            (Sequence[Choice]):
                 A sequence (such as a list or tuple) of Choice objects

        """
        return self._choices

    def to_consumptions(self) -> Sequence[Consumption]:
        """
        Converts detailed usage metrics into a sequence of `Consumption` objects.
        This function takes no arguments and generates a sequence of `Consumption` objects, each representing different aspects of usage metrics such as number
        of calls, tokens used in prompts, tokens generated in completions, and total tokens used. Every `Consumption` object contains the quantity of the metric,
        the unit in which it is measured (e.g., 'call' or 'token'), and a string describing the kind of usage it represents, which includes the model name and the specific usage type.
        
        Returns:
            (Sequence[Consumption]):
                 A list containing `Consumption` objects for each usage metric.

        """
        return [
            Consumption(1, "call", f"{self.model}"),
            Consumption(self.usage.prompt_tokens, "token", f"{self.model}:prompt_tokens"),
            Consumption(self.usage.completion_tokens, "token", f"{self.model}:completion_tokens"),
            Consumption(self.usage.total_tokens, "token", f"{self.model}:total_tokens"),
        ]

    @staticmethod
    def from_dict(obj: Any) -> OpenAIChatCompletionsResult:
        """
        Converts a dictionary to an instance of OpenAIChatCompletionsResult.
        This static method transforms a dictionary object representing an OpenAI Chat
        Completions API response into an instance of the OpenAIChatCompletionsResult class.
        
        Args:
            obj (Any):
                 A dictionary object containing keys and values corresponding to
                the attributes of OpenAIChatCompletionsResult. Keys in the 'obj' include 'id',
                'object', 'created', 'model', 'choices', and 'usage'.
        
        Returns:
            (OpenAIChatCompletionsResult):
                 An instance of OpenAIChatCompletionsResult containing
                the data from the provided dictionary.
        
        Raises:
            ValueError:
                 An error occurs if the dictionary doesn't have the required keys
                or if any of the required keys have values that cannot be converted to the
                expected type.
        
        Note:
            This method assumes that 'choices' is a list of dictionaries that can be
            correctly converted using Choice.from_dict static method, and 'usage' is a
            dictionary that can be correctly converted using Usage.from_dict static method.
            Both Choice and Usage should have corresponding from_dict static methods
            implemented. If any of the from_dict conversions fail, the method will propagate
            the exception.
            

        """
        _id = str(obj.get("id"))
        _object = str(obj.get("object"))
        _created = int(obj.get("created"))
        _model = str(obj.get("model"))
        _choices = [Choice.from_dict(y) for y in obj.get("choices")]
        _usage = Usage.from_dict(obj.get("usage"))
        return OpenAIChatCompletionsResult(_id, _object, _created, _model, _choices, _usage)


class OpenAIChatCompletionsModel(LLMBase):
    """
    A class representing the OpenAI Chat Completions model which interacts with a language model to provide
    completions for chat-based inputs. It extends `LLMBase` to make use of specific configurations and providers for
    sending requests and processing the received outcomes.
    
    Attributes:
        config (LLMConfigurationBase):
             An instance of a configuration object containing parameters for the language
            model and API interaction.
        _provider (Provider):
             An instance of a provider that abstracts the API call mechanism.
        token_counter (LLMessageTokenCounterBase, optional):
             An object to keep track of token usage across messages.
        name (str, optional):
             An optional name given to the instance for identification, debugging, or logging.
    
    Methods:
        _post_chat_request(context:
             LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
            A method to send a chat request to the language model and retrieve the result. It processes a sequence of
            messages along with optional keyword arguments, constructs a payload, and logs the request and response
            details.
        _post_request(payload) -> OpenAIChatCompletionsResult:
            A method that abstracts the HTTP request sending process. It receives a payload, makes an API call through
            the provider instance, and returns an `OpenAIChatCompletionsResult` object.
    
    Raises:
        LLMCallException:
             An exception is raised when the API call response status is not OK (200), containing
            the status code, response text, and the name of the model instance.

    """

    def __init__(
        self,
        config: LLMConfigurationBase,
        provider: Provider,
        token_counter: Optional[LLMessageTokenCounterBase],
        name: Optional[str] = None,
    ):
        """
        Initializes a new instance of the class with the given configuration, provider, and token counter.
        
        Args:
            config (LLMConfigurationBase):
                 The configuration object used to set up the class.
            provider (Provider):
                 The provider object that offers services or resources to the class.
            token_counter (Optional[LLMessageTokenCounterBase]):
                 An optional token counter object for tracking message tokens.
            name (Optional[str]):
                 An optional name for the instance. Defaults to None if not provided.
        
        Raises:
            TypeError:
                 If the given arguments are of incorrect type.
        
        Note:
            The super().__init__ call within this method points to the base class initializer and may pass additional parameters such as token_counter and name to it.

        """
        super().__init__(token_counter, name)
        self.config = config
        self._provider = provider

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Generates and sends a chat request to an LLM service, then processes the response.
        This function assembles a payload for a chat request from provided messages and additional keyword
        arguments. It performs the HTTP request using a private method and logs the process. Once the
        response is received, it extracts chat completion results and constructs an LLMResult object.
        
        Args:
            context (LLMContext):
                 The context object containing configuration and logger.
            messages (Sequence[LLMMessage]):
                 A sequence of messages that will form the chat history
                for the LLM to consider in generating its response.
            **kwargs (Any):
                 Additional keyword arguments that may be required for the payload.
        
        Returns:
            (LLMResult):
                 An object containing the choices and consumptions from the LLM response.
        
        Note:
            This method is for internal use and uses a private method to post the HTTP request. The
            logger is used to output debug information regarding the request and response.

        """
        payload = self.config.build_default_payload()
        payload["messages"] = [message.dict() for message in messages]
        for key, value in kwargs.items():
            payload[key] = value

        context.logger.debug(f'message="Sending chat GPT completions request to {self._name}" payload="{payload}"')
        r = self._post_request(payload)
        context.logger.debug(
            f'message="Got chat GPT completions result from {self._name}" id="{r.id}" model="{r.model}" {r.usage}'
        )
        return LLMResult(choices=[c.message.content for c in r.choices], consumptions=r.to_consumptions())

    def _post_request(self, payload) -> OpenAIChatCompletionsResult:
        """
        Sends a POST request with the given payload to the provider's API and returns the result.
        This method sends a payload to the provider's API endpoint, which is expected to result in a chat completion.
        If the HTTP response status code is not 200 (OK), it raises an LLMCallException with details about the failure.
        Otherwise, it creates an instance of OpenAIChatCompletionsResult from the response JSON data and returns it.
        
        Args:
            payload (dict):
                 The payload data to send in the POST request.
        
        Returns:
            (OpenAIChatCompletionsResult):
                 An instance constructed from the API response data.
        
        Raises:
            LLMCallException:
                 If the response status code is not OK (200), including the status code and error message
                from the response, as well as the provider's name.

        """
        response = self._provider.__call__(payload)
        if response.status_code != httpx.codes.OK:
            raise LLMCallException(response.status_code, response.text, self._name)

        return OpenAIChatCompletionsResult.from_dict(response.json())
