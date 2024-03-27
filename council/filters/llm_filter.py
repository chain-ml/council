"""

Module llm_filter

This module defines classes and functionalities for filtering the output of Language Model
(LLM) based on specific criteria. It primarily contains the FilterResult class, which holds
the results of the filtering process, and the LLMFilter class, which is responsible for
executing the filtering using a provided LLM model. The module leverages types defined in
council.contexts, council.filters, and council.llm modules.

Classes:
    FilterResult: A class that encapsulates the result of a filtering operation, including the index
    of the message, the flag indicating whether it was filtered, and a justification for the decision.

    LLMFilter: A class that extends FilterBase to implement the logic for filtering text using a
    given LLM. It manages communication with the LLM, sends requests, and handles responses.

    LLMParsingException: Exception class defined in "council.llm.llm_answer" module that signals issues with
    parsing the output of the LLM used by LLMFilter.

Functions:
    FilterResult.is_filtered: A property indicating whether the message was filtered.

    FilterResult.index: A property that returns the index of the message in question.

    FilterResult.justification: A property that provides the justification behind the message's
    filtering decision.

    LLMFilter.__init__: Constructor that initializes the LLMFilter with a specific LLM and optionally
    a list of criteria to filter on.

    LLMFilter._execute: Protected method that processes all messages through the LLM model and
    filters them based on the given criteria and LLM's responses.

    LLMFilter._handle_error: Static method that logs and handles any exceptions raised during
    filtering, producing new LLM messages for further clarification as needed.

    LLMFilter._parse_response: Protected method that parses the LLM's response into FilterResults
    and decides which messages should be kept based on the filtering criteria.

    LLMFilter._build_llm_messages: Protected method that generates the initial LLM messages for the
    filtering task, including the setup context and user's message.

    LLMFilter._parse_line: Static method that parses a line of LLM's response into a FilterResult,
    wrapped in an Option type to handle potentially absent data safely.

    LLMFilter._build_user_message: Protected method that constructs the user query message for the
    LLM based on the messages to be filtered and the current filter criteria.

    LLMFilter._build_system_message: Protected method that creates the system setup message for the
    LLM detailing the role, instructions, and formatting requirements for the task.



"""

from typing import List, Optional

from council.contexts import AgentContext, ScoredChatMessage, ContextBase
from council.filters import FilterBase, FilterException
from council.llm import LLMBase, MonitoredLLM, llm_property, LLMAnswer, LLMMessage
from council.llm.llm_answer import LLMParsingException
from council.utils import Option


class FilterResult:
    """
    A class representing the result of a filter operation on a message, indicating whether the message was filtered, its index in the sequence and the justification for the result.
    
    Attributes:
        _index (int):
             The index of the message in a sequence or list.
        _filtered (bool):
             A boolean indicating whether the message was filtered (True) or not (False).
        _justification (str):
             A description or reason explaining why the message was filtered.
    
    Methods:
        is_filtered:
             A property that returns whether the message was filtered.
        index:
             A property that returns the index of the message.
        justification:
             A property that returns the justification for filtering the message.
        __str__:
             Returns a string representation indicating whether the message was filtered, its index, and the justification.
        

    """
    def __init__(self, index: int, is_filtered: bool, justification: str):
        """
        Initializes a new instance with specified index, filter status, and justification.
        
        Args:
            index (int):
                 The index or position of the item.
            is_filtered (bool):
                 A flag indicating whether the item is filtered.
            justification (str):
                 A textual explanation for why the item is filtered or not.
        
        Attributes:
            _filtered (bool):
                 Stores the filter status of the item.
            _index (int):
                 Stores the index or position of the item.
            _justification (str):
                 Stores the justification for the item's filter status.

        """
        self._filtered = is_filtered
        self._index = index
        self._justification = justification

    @llm_property
    def is_filtered(self) -> bool:
        """
        
        Returns a boolean value indicating whether the object is in a filtered state or not.
            This method checks the private '_filtered' attribute of the object, which is a boolean value that signifies whether the object has been
            filtered. The method returns the value of this attribute, thus allowing the caller to know if the object is in a filtered state.
        
        Returns:
            (bool):
                 True if the object is filtered, False otherwise.
            

        """
        return self._filtered

    @llm_property
    def index(self) -> int:
        """
        
        Returns the index of the current object.
        
        Returns:
            (int):
                 The index value stored within the object's _index attribute.

        """
        return self._index

    @llm_property
    def justification(self) -> str:
        """
        
        Returns the justification property of the object.
            This method acts as a property getter for the private attribute '_justification' that exists within an object. This attribute typically represents the reasoning or explanation behind a certain decision or value assigned to the object. Since direct access to private attributes is discouraged and against the practice in OOP (Object-Oriented Programming), this method provides a secure way to obtain the justification value.
        
        Returns:
            (str):
                 The justification text associated with the object.

        """
        return self._justification

    def __str__(self):
        """
        Converts the object to its informal string representation that includes filtered status and justification.
        
        Returns:
            (str):
                 An informal and readable string representation of the Message object indicating
                whether it is filtered, along with the justification for its status.

        """
        t = " " if self._filtered else " not "
        return f"Message {self._index} is{t}filtered with the justification: {self._justification}"


class LLMFilter(FilterBase):
    """
    A class for filtering messages based on criteria using a language model (LLM) interface.
    This class extends the functionality of a base filter class and incorporates a
    language model as a decision-making tool for whether messages should be filtered.
    In its operation, the class receives messages to be filtered and uses the LLM
    to process and determine which of those messages satisfy the filtering conditions.
    
    Attributes:
        _llm (LLMBase):
             An instance of a language learning model which is monitored.
        _llm_answer (LLMAnswer):
             An object used for interpreting the LLM's output.
        _filter_on (Optional[List[str]]):
             A list of criteria to filter the messages on.
        _retry (int):
             The number of retries in case of an error during message evaluation.
    
    Methods:
        _execute(context:
             AgentContext) -> List[ScoredChatMessage]: Executes the filter processing using the LLM.
        _handle_error(e:
             Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]: Handles error cases during processing.
        _parse_response(context:
             ContextBase, response: str, messages: List[ScoredChatMessage]) -> List[ScoredChatMessage]: Parses the LLM's response and filters messages.
        _build_llm_messages(messages:
             List[ScoredChatMessage]) -> List[LLMMessage]: Constructs messages to prompt the LLM for processing.
        _parse_line(line:
             str) -> Option[FilterResult]: Parses individual lines in the LLM's response.
        _build_user_message(messages:
             List[ScoredChatMessage]) -> LLMMessage: Builds a user message prompt for the LLM.
        _build_system_message() -> LLMMessage:
             Builds a system message prompt indicating the LLM's role and instructions.

    """

    def __init__(self, llm: LLMBase, filter_on: Optional[List[str]] = None):
        """
        Initializes a new instance of the class that handles interactions with an LLMBase object.
        This method sets up the necessary components to monitor and interact with a language model, represented by the LLMBase type.
        It registers the language model as a monitored entity and also initializes an LLMAnswer object, which is used to handle responses based on a filter result schema.
        Optionally, a filter can be applied to the LLM's outputs through the 'filter_on' parameter.
        Upon initialization, the number of retry attempts for interactions with the LLM is set to 3 by default.
        
        Args:
            llm (LLMBase):
                 The language model being interacted with and monitored.
            filter_on (Optional[List[str]], optional):
                 A list of string criteria used to filter outputs from the language model. Defaults to None, which means no filter is applied.

        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._llm_answer = LLMAnswer(FilterResult)
        self._filter_on = filter_on or []
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Parses a YAML formatted string into a list of dictionaries and checks for necessary keys.
        This method parses a YAML string, which contains a list of elements, into a list of dictionaries. Each element in the YAML string is expected to be
        a dictionary itself. After parsing, it checks each dictionary for the presence of required keys as specified by `self._properties`. If any
        of the required keys are missing from an item, it raises an LLMParsingException.
        
        Args:
            bloc (str):
                 A YAML formatted string representing a list of dictionaries.
        
        Returns:
            (List[Dict[str, Any]]):
                 A list of dictionaries, each representing an item from the YAML list with all required keys present.
        
        Raises:
            LLMParsingException:
                 If any of the parsed dictionaries is missing one or more of the required keys.

        """
        all_eval_results = list(context.evaluation)
        if all_eval_results is None:
            return []

        if len(self._filter_on) == 0:
            return all_eval_results

        retry = self._retry
        messages = self._build_llm_messages(all_eval_results)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            retry -= 1
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                return self._parse_response(context, response, all_eval_results)
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except FilterException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise FilterException("LLMFilter failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        """
        Handles errors by logging and creating messages to assist the user.
        This static method is used to handle exceptions that occur within the system. It logs the
        exception information and returns a list of messages intended for the assistant to display.
        The assistant message provides a user-friendly explanation, while the user message offers
        a fix by displaying the original exception details.
        
        Args:
            e (Exception):
                 The exception that was raised.
            assistant_message (str):
                 A message that provides context or help to the assistant.
            context (ContextBase):
                 The context in which the error occurred, with a logger.
        
        Returns:
            (List[LLMMessage]):
                 A list containing the assistant message and a user message
                with details about the exception that occurred.

        """
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _parse_response(
        self, context: ContextBase, response: str, messages: List[ScoredChatMessage]
    ) -> List[ScoredChatMessage]:
        """
        __parse_response(self, context: ContextBase, response: str, messages: List[ScoredChatMessage]) -> List[ScoredChatMessage]
        Parses the given response string into a list of scored chat messages after applying filters.
        This method takes a response string, which is expected to follow a specific format, and a list of scored chat messages. The method processes the response by splitting it into lines and parsing each line. Each parsed line is unwrapped to extract valid answers that are not filtered out. The method returns the list of messages that are allowed by the filters. If parsing fails or if there are missing filter responses for any answer, specific exceptions are raised.
        
        Args:
            context (ContextBase):
                 The context associated with the response parsing, providing logging and other contextual information.
            response (str):
                 The response string from a previous operation that needs parsing.
            messages (List[ScoredChatMessage]):
                 A list of scored chat messages that are expected to be filtered.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of messages that have passed the filtering process.
        
        Raises:
            LLMParsingException:
                 If no answers in the response could be parsed or if they do not follow the expected format.
            FilterException:
                 If not all messages have corresponding filter responses in the parsed response.

        """
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        answers = [r.unwrap() for r in parsed if r.is_some()]
        if len(answers) == 0:
            raise LLMParsingException("None of your answer could be parsed. Follow exactly formatting instructions.")

        messages_to_keep = []
        missing = []
        for idx, message in enumerate(messages):
            try:
                answer = next(filter(lambda item: item.index == (idx + 1), answers))
                if not answer.is_filtered:
                    messages_to_keep.append(message)
                context.logger.debug(f"{answer} for {message.message}")
            except StopIteration:
                missing.append(idx)

        if len(missing) > 0:
            raise FilterException(
                f"Please evaluate ALL {len(messages)} answers. Missing filter responses for {missing} answers."
            )

        return messages_to_keep

    def _build_llm_messages(self, messages: List[ScoredChatMessage]) -> List[LLMMessage]:
        """
        Builds a list of low-level messages (LLM) for processing from scored chat messages.
        This method prepares messages for a lower-level messaging or processing system. It constructs
        a list which includes a system-generated message followed by a user-generated message based on
        the given scored chat messages.
        
        Args:
            messages (List[ScoredChatMessage]):
                 A list of ScoredChatMessage objects which contain
                information about the messages along with their respective scores.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage objects which includes a system message followed
                by a user message constructed from the scored chat messages.

        """
        return [self._build_system_message(), self._build_user_message(messages)]

    def _parse_line(self, line: str) -> Option[FilterResult]:
        """
        Parses a line and attempts to convert it into a `FilterResult` object wrapped in an `Option` type.
        This method checks if the line contains a specific field separator. If the field
        separator is found, it attempts to convert the line into an `Option` containing
        a `FilterResult` instance, using the conversion logic defined in `_llm_answer.to_object`.
        If the conversion is successful, the resulting `FilterResult` is wrapped as `Option.some`.
        Otherwise, if the field separator is not in the line, or the line cannot be
        converted to `FilterResult`, an `Option.none` is returned, representing the absence of a
        value.
        
        Args:
            line (str):
                 The input string to be parsed.
        
        Returns:
            (Option[FilterResult]):
                 An `Option` object which either contains a `FilterResult`
                instance if parsing was successful, or is empty (`Option.none`) if parsing
                failed or if the field separator was not present in the line.

        """
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[FilterResult] = self._llm_answer.to_object(line)
        return Option(cs)

    def _build_user_message(self, messages: List[ScoredChatMessage]) -> LLMMessage:
        """
        Constructs a user message for the Low-level Messaging system based on a list of scored chat messages.
        This method organizes the given messages and any active filters into a coherent string to be used as a prompt in the Low-level Messaging (LLM) system. It formats the scored chat messages with an enumerated list, and if any filters are set, it includes those as well. The resulting string is used to create an LLMMessage object that represents a message from a user.
        
        Args:
            messages (List[ScoredChatMessage]):
                 A list of scored chat messages, each typically containing a score that indicates the relevance or quality of the message.
        
        Returns:
            (LLMMessage):
                 An instance of LLMMessage containing the user prompt composed of the listed answers and any applicable filters.

        """
        prompt_answers = "\n".join(
            f"- answer #{index + 1} is: {message.message}" for index, message in enumerate(messages)
        )
        filters = "\n".join(f"- {filter}" for filter in self._filter_on)

        lines = [
            "\nFILTERS",
            filters,
            "\nPlease filter or not the following answers according to your instructions:",
            prompt_answers,
        ]
        prompt = "\n".join(lines)
        return LLMMessage.user_message(prompt)

    def _build_system_message(self) -> LLMMessage:
        """
        Builds a system message for the AI system based on the task description and instructions.
        This message is constructed by joining a series of predefined strings that describe
        the role, the instructions for decision-making, the evaluation criteria for answers,
        formatting guidelines, and any additional information needed to generate a system prompt.
        The purpose is to create a structured message that correctly guides the AI in
        its response generation and ensures it follows the intended format and evaluation protocol.
        
        Returns:
            (LLMMessage):
                 An object containing the system message structured as a prompt.
                This message is used to communicate with the AI system and instruct it on how to
                process and evaluate the responses in accordance with the specified role and instructions.
            

        """
        task_description = [
            "\n# ROLE",
            "You are an judge, with a large breadth of knowledge.",
            "You are deciding with objectivity if some answers from different Specialists need to be filtered.",
            "\n# INSTRUCTIONS",
            "1. Give your response with TRUE or FALSE",
            "2. Evaluate carefully and fairly the proposed answer.",
            "3. Ignore how assertive the answer is, only content accuracy count."
            "4. Consider only the Specialist's answer and ignore its index.",
            "5. Ensure to be consistent, identical answers must have the same response.",
            "\n# FORMATTING",
            "1. The list of given answers is formatted precisely as:",
            "- answer #{index} is: {Specialist's answer or EMPTY if no answer}",
            "2. For each given answer, format your response precisely as:",
            self._llm_answer.to_prompt(),
        ]
        prompt = "\n".join(task_description)
        return LLMMessage.system_message(prompt)
