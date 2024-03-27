"""

Module that provides an implementation of a skill that uses a Language Model (LLM) for generating responses in a chat system.

Classes:
    ReturnMessages(Protocol): An interface for a function that takes a `SkillContext` and returns a list of `LLMMessage`.

    PromptToMessages: A class that converts prompts to `LLMMessage` instances meant to be sent to the system or user.

    LLMSkill(SkillBase): A skill that uses Language Models to generate responses, with methods for executing the skill and handling LLM.

Functions:
    get_chat_history(context: SkillContext) -> List[LLMMessage]: Retrieves the chat history from the skill context as a list of LLMMessage instances.

    get_last_messages(context: SkillContext) -> List[LLMMessage]: Retrieves the last message from the skill context, if available, or falls back to retrieving the chat history.

This module utilizes Types from other modules, including `SkillContext`, `ChatMessage`, `LLMBase`, `LLMMessage`, `MonitoredLLM`, `PromptBuilder`, and `SkillBase`. The `LLMSkill` class is the main entry point in this module for interfacing with an LLM within the context of executing a skill.


"""
from typing import List, Protocol

from council.contexts import SkillContext, ChatMessage
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.prompt import PromptBuilder
from council.skills import SkillBase


class ReturnMessages(Protocol):
    """
    A protocol defining a callable interface that, when implemented, is expected to receive a SkillContext object and return a list of LLMMessage objects.

    Attributes:
        context (SkillContext):
             An instance of SkillContext that provides the necessary context for generating the messages.

    Returns:
        (List[LLMMessage]):
             A list of LLMMessage objects that have been generated based on the provided SkillContext.

    """

    def __call__(self, context: SkillContext) -> List[LLMMessage]:
        """
        Magic method to make the object callable. Processes the given context and returns a list of LLMMessage objects.

        Args:
            context (SkillContext):
                 The context in which the skill is being executed. This includes any relevant data and state required for the method to perform its logic.

        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage objects which are constructed based on the input context and the outcome of processing the skill logic.


        """
        ...


def get_chat_history(context: SkillContext) -> List[LLMMessage]:
    # Convert chat's history and give it to the inner llm
    """
    Retrieves the chat history from a given skill context and converts it to a list of LLMMessage objects.
    
    Args:
        context (SkillContext):
             The skill context object containing the chat history
            to be retrieved and transformed.
    
    Returns:
        (List[LLMMessage]):
             A list of LLMMessage objects representing the converted chat history messages.

    """
    return LLMMessage.from_chat_messages(context.chat_history.messages)


def get_last_messages(context: SkillContext) -> List[LLMMessage]:
    """
    Retrieves the list of last user messages from the skill context.
    This function examines the given skill context and aims to return a list of the last LLMMessage objects sent by a user. If an 'iteration' context is present and has data, it unwraps it to create a message from the value. If no such context is available, it attempts to obtain the last message from the current context. If there are no last messages, it defaults to fetching the full chat history.
    
    Args:
        context (SkillContext):
             The context of the skill which contains information about the chat history and current interaction.
    
    Returns:
        (List[LLMMessage]):
             A list containing the last LLMMessage sent by the user. This list may contain a single message, or multiple messages if they are fetched from the chat history.
        

    """
    if context.iteration.is_some():
        it_ctxt = context.iteration.unwrap()
        msg = LLMMessage.user_message(it_ctxt.value)
        return [msg]
    last_message = context.current.try_last_message
    if last_message.is_none():
        return get_chat_history(context)
    msg = LLMMessage.user_message(last_message.unwrap().message)
    return [msg]


class PromptToMessages:
    """
    A class that converts prompts into system or user messages using a PromptBuilder.
    The PromptToMessages class is responsible for transforming a given prompt into either a system-generated message or a
    user-generated message. It utilizes a PromptBuilder object to apply context and generate the appropriate message format.
    The class is intended to be used in a larger system where messaging between a user and the system is critical, such as
    in a chatbot or other conversational AI interface.
    
    Attributes:
        _builder (PromptBuilder):
             An instance of PromptBuilder used for message generation.
    
    Methods:
        to_system_message(context:
             SkillContext) -> List[LLMMessage]:
            Takes a SkillContext object and generates a system message based the current state of the context.
            The message is then wrapped in a LLMMessage object and returned as a list containing one message.
        to_user_message(context:
             SkillContext) -> List[LLMMessage]:
            Generates a user message based on the SkillContext by applying the prompt_builder.
            Similar to to_system_message, the message is encapsulated in a LLMMessage object and returned in a list.
        

    """
    def __init__(self, prompt_builder: PromptBuilder):
        """
        Initializes the object with a PromptBuilder instance for building prompts.
        
        Args:
            prompt_builder (PromptBuilder):
                 An instance of PromptBuilder used to construct prompts.

        """
        self._builder = prompt_builder

    def to_system_message(self, context: SkillContext) -> List[LLMMessage]:
        """
        Generates a system message based on the given SkillContext and formats it into a list of LLMMessage objects.
        This function processes the context using the internal message builder to create a message string. Then it logs the message as a debug statement and wraps the message in a LLMMessage object designed for system messages. The resulting LLMMessage object gets returned as a single-element list.
        
        Args:
            context (SkillContext):
                 The context object that provides relevant data and logger to build the system message.
        
        Returns:
            (List[LLMMessage]):
                 A list containing a single LLMMessage object as a system message.

        """
        msg = self._builder.apply(context)
        context.logger.debug(f'prompt="{msg}')
        return [LLMMessage.system_message(msg)]

    def to_user_message(self, context: SkillContext) -> List[LLMMessage]:
        """
        Converts the given context into a user-readable message format.
        This method uses the internal message builder to create a message from the provided SkillContext. It logs the generated message
        at the debug level and encapsulates it into a list as a LLMMessage object designed to be understood by the user.
        
        Args:
            context (SkillContext):
                 The context from which the user message is generated. It includes all necessary information
                that the message builder might require to compose the message.
        
        Returns:
            (List[LLMMessage]):
                 A list containing the generated user message as a LLMMessage instance.
            

        """
        msg = self._builder.apply(context)
        context.logger.debug(f'prompt="{msg}')
        return [LLMMessage.user_message(msg)]


class LLMSkill(SkillBase):
    """
    A skill that encapsulates the interaction with a language model.
    This class inherits from `SkillBase` and is designed to provide a high-level interface for interacting with a language model. The class also monitors the language model's state and provides mechanisms to construct prompts and parse responses.
    
    Attributes:
        _llm:
             A monitored instance of a language model.
        _context_messages:
             A callable that retrieves history messages from the context.
        _builder:
             An instance of `PromptBuilder` for constructing system prompts.
        Properties:
        llm:
             The inner language model accessed through the monitored instance.
    
    Methods:
        __init__:
             Initializes a new instance of the `LLMSkill` class.
        execute:
             Processes the incoming context to construct a prompt, gets the response from the language model and constructs the output message.

    """

    def __init__(
        self,
        llm: LLMBase,
        name: str = "LLMSkill",
        system_prompt: str = "",
        context_messages: ReturnMessages = get_last_messages,
    ):
        """
        Initialize a new instance of the `LLMSkill` class or its subclass.
        This constructor takes in an `LLMBase` object, which is the language model to interact with, optionally
        a name for the skill, a system prompt to pre-populate the prompt builder with, and a customizable
        context_messages function to determine how to fetch messages from the context.
        
        Parameters:
            llm (LLMBase):
                 An instance of a language model that will be used by this skill.
            name (str):
                 An optional name for the skill instance. Defaults to 'LLMSkill'.
            system_prompt (str):
                 An initial prompt to pre-populate the prompt builder with. Defaults to an empty string.
            context_messages (ReturnMessages):
                 A callable to determine the context messages fetching strategy.
                Defaults to the `get_last_messages` function.
            Side Effects:
                - The provided `LLMBase` instance will be wrapped in a `MonitoredLLM` and registered as a monitor.
                - The `PromptBuilder` will be instantiated with the given system prompt.
            

        """

        super().__init__(name=name)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._context_messages = context_messages
        self._builder = PromptBuilder(system_prompt)

    @property
    def llm(self) -> LLMBase:
        """
        Property that gets the underlying `LLMBase` instance.
        This property provides access to the inner `LLMBase` instance that is wrapped by this object.
        It allows the user to interact directly with the low-level model, offering more control
        or customization when needed. The actual implementation details of `_llm` are encapsulated
        and should not be a concern of the user of this property.
        
        Returns:
            (LLMBase):
                 The `LLMBase` instance that is encapsulated by this object.
            

        """
        return self._llm.inner

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes the skill within the provided context and returns the chat message response from the Language Model (LLM).
        This method collects historical chat messages, creates a system prompt based on the context by applying a builder, and combines them to send a request to the LLM. Upon receiving a response from the LLM, it checks if there are any choices. If there are no choices in the response (indicating no response from the LLM), an error message is built and returned. Otherwise, it adds a consumption to the context budget and returns a success message with the LLM's first choice and the complete LLM response.
        
        Args:
            context (SkillContext):
                 The context of the skill execution, containing relevant data and state.
        
        Returns:
            (ChatMessage):
                 The chat message to be returned after execution. This could either be an error message if there was no response from the LLM, or a success message encapsulating the LLM's response.
        
        Raises:
            It does not explicitly raise exceptions but might return an error message if issues occur with the LLM response.

        """

        history_messages = self._context_messages(context)
        system_prompt = LLMMessage.system_message(self._builder.apply(context))
        messages = [system_prompt, *history_messages]
        llm_response = self._llm.post_chat_request(context, messages=messages)
        if len(llm_response.choices) < 1:
            return self.build_error_message(message="no response")

        context.budget.add_consumption(1, "call", "LLMSkill")

        return self.build_success_message(message=llm_response.first_choice, data=llm_response)
