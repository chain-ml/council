"""

Module provides functionality for evaluating messages using Language Model (LLM).

This module contains the LLMEvaluator class, which is a subclass of EvaluatorBase,
and uses the MonitoredLLM class to send queries to a language model and interpret
the results.

The LLMEvaluator class manages the evaluation process of chat messages based on
grading criteria provided by a language model. It supports retrying evaluation
attempts when necessary and handles errors during the interpretation of the
language model's responses.

Classes:
    LLMEvaluator: An evaluator with retry logic for interpreting language model
                  (LLM) responses and grading messages accordingly.

Exceptions:
    LLMParsingException: Custom exception for handling parsing errors.

Attributes:
    SpecialistGrade — A data class used within LLMEvaluator for associating
                     chat message indices with their respective grades and
                     justifications.


"""

from typing import List, Optional

from council.contexts import AgentContext, ChatMessage, ScoredChatMessage, ContextBase
from council.evaluators import EvaluatorBase, EvaluatorException
from council.llm import LLMBase, MonitoredLLM, llm_property, LLMAnswer, LLMMessage
from council.llm.llm_answer import LLMParsingException, llm_class_validator
from council.utils import Option


class SpecialistGrade:
    """
    A class that encapsulates the grading details of a specialist.
    This class holds information about a specialist's grade, the associated index, and the justification for that grade. It provides property methods to access these details and includes a validation method to ensure the grade is within a valid range.
    
    Attributes:
        _grade (float):
             A floating-point number representing the specialist's grade.
        _index (int):
             An integer representing the specialist's index.
        _justification (str):
             A string stating the reason for the given grade.
    
    Methods:
        __init__:
             Initializes a new instance of SpecialistGrade.
        grade:
             Property to get the specialist's grade.
        index:
             Property to get the specialist's index.
        justification:
             Property to get the justification for the specialist's grade.
        __str__:
             Returns a formatted string representation of the SpecialistGrade instance.
        validate:
             Validates that the grade is within a specified range, raising an exception if it is not.
    
    Raises:
        LLMParsingException:
             An error indicating that the given grade is outside the acceptable range.

    """
    def __init__(self, index: int, grade: float, justification: str):
        """
        Initialize a new instance of the class with specified index, grade, and justification.
        This method sets the initial state of the object by assigning the values provided
        to the corresponding instance variables. It is automatically called when a new
        object is created from the class where this method is defined.
        
        Args:
            index (int):
                 An integer representing a unique identifier for the instance.
            grade (float):
                 A float representing the grade or score attributed to
                the instance.
            justification (str):
                 A string containing explanatory text or reasoning
                behind the grade assigned.
        
        Raises:
            TypeError:
                 If the types of the arguments do not match the expected
                (e.g., `index` is not an integer, `grade` is not a float,
                or `justification` is not a string).
            

        """
        self._grade = grade
        self._index = index
        self._justification = justification

    @llm_property
    def grade(self) -> float:
        """
        
        Returns the grade for the instance as a float.
            This method simply returns the value of the private _grade attribute associated with the
            instance of the class. It is a getter method that provides a public interface to access
            the internally stored grade value.
        
        Returns:
            (float):
                 The grade of the instance.

        """
        return self._grade

    @llm_property
    def index(self) -> int:
        """
        
        Returns the index attribute of the object.
            This property method provides access to the _index attribute, typically representing the object's position
            in a sequence or collection. The index is expected to be an integer.
        
        Returns:
            (int):
                 The integer value of the _index attribute.
            

        """
        return self._index

    @llm_property
    def justification(self) -> str:
        """
        Gets the justification for an object.
        
        Returns:
            (str):
                 The justification as a string.

        """
        return self._justification

    def __str__(self):
        """
        Return a string representation of the message object.
        
        This method overrides the default `__str__` method to provide a meaningful string
        representation for the message object which includes its index, grade, and justification.
        
        Returns:
            (str):
                 The string representation of the message object formatted to include index,
                grade, and a justification.
            

        """
        return f"Message `{self._index}` graded `{self._grade}` with the justification: `{self._justification}`"

    @llm_class_validator
    def validate(self):
        """
        
        Raises an LLMParsingException if the grade attribute of the instance is not within the valid range.
            This method checks whether the `self._grade` attribute of an instance falls within the valid range of 0.0 to 10.0. If the grade is outside this range, the method raises an LLMParsingException with a message indicating the invalidity of the grade value.
        
        Raises:
            LLMParsingException:
                 An error indicating that the grade value is out of the acceptable range (0.0 - 10.0).

        """
        if self._grade < 0.0 or self._grade > 10.0:
            raise LLMParsingException(f"Grade `{self._grade}` is invalid, value must be between 0.0 and 10.0")


class LLMEvaluator(EvaluatorBase):
    """
    A specialized evaluator class for scoring chat messages using a language model (LLM).
    This class extends from EvaluatorBase and specializes in evaluating responses from a language learning model
    (LLM) using a predefined grading scheme. Interaction with the LLM is done through the MonitoredLLM class
    which is registered as a monitor within LLMEvaluator.
    
    Attributes:
        _llm (MonitoredLLM):
             A monitored instance of an LLM that evaluates and grades the responses.
        _llm_answer (LLMAnswer):
             A container holding the response format and converting it to/from a SpecialistGrade
            object.
        _retry (int):
             The number of times the evaluation process should be retried in case of failure or errors.
    
    Methods:
        __init__(llm:
             LLMBase): Initializes the LLMEvaluator with a specific instance of an LLM.
        _execute(context:
             AgentContext): Handles the evaluation workflow by sending requests to the LLM and processing
            the responses.
        _handle_error(e:
             Exception, assistant_message: str, context: ContextBase): Generates a message in the case of an
            encountered exception.
        _parse_response(context:
             ContextBase, response: str, chain_results: List[ChatMessage]): Parses the LLM's
            graded responses, associates them with chat messages, and returns a list of ScoredChatMessages.
        _build_llm_messages(query:
             ChatMessage, skill_messages: List[ChatMessage]): Constructs LLM-specific messages
            for submission based on the chat history and skill messages.
        _parse_line(line:
             str): Processes a single line of response from the LLM, extracting the grading if properly
            formatted.
        _build_user_message(query:
             str, answers: list[str]): Creates a user message with the query and associated
            answers to be graded by the LLM.
        _build_system_message():
             Constructs the message containing the grading task description and instructions for
            the LLM.

    """

    def __init__(self, llm: LLMBase):
        """
        Constructor to initialize a new instance of the wrapping class.
        This method initializes the class instance with a provided LLMBase object, registers the LLMBase object to be monitored, creates an LLMAnswer instance for SpecialistGrade schema validation, and sets a default retry value.
        
        Args:
            llm (LLMBase):
                 An instance of LLMBase or any of its subclasses to be used and monitored in this class.
        
        Attributes:
            _llm (MonitoredLLM):
                 A wrapped and monitored instance of LLMBase registered with a specific identifier.
            _llm_answer (LLMAnswer):
                 An LLMAnswer instance for parsing and validating responses according to the SpecialistGrade schema.
            _retry (int):
                 The number of retries allowed for specific operations, initialized to a default value (often used in case of failures).

        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._llm_answer = LLMAnswer(SpecialistGrade)
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Execute the language model's chat request and process its results.
        This function orchestrates the execution of a chat request to a language
        model using the provided context. It compiles the query from the last user
        message and the last messages from the context chains, builds LLM messages by
        combining them, and posts a chat request. If the parsing of the response
        fails or raises an exception, it handles the error and retries the request
        until the maximum number of retries is reached.
        
        Args:
            context (AgentContext):
                 The context providing chat history, chain state, and other necessary information.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of scored chat messages that represent the parsed response of the language model.
        
        Raises:
            LLMParsingException:
                 An exception for handling errors during the parsing of the language model's response.
            EvaluatorException:
                 Raised if the evaluator fails to execute after all retries or if there is an error during response parsing or exception handling.
            

        """
        query = context.chat_history.try_last_user_message.unwrap()
        chain_results = [
            chain_messages.try_last_message.unwrap()
            for chain_messages in context.chains
            if chain_messages.try_last_message.is_some()
        ]

        retry = self._retry
        messages = self._build_llm_messages(query, chain_results)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            retry -= 1
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                parse_response = self._parse_response(context, response, chain_results)
                return parse_response
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except EvaluatorException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise EvaluatorException("LLMEvaluator failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        """
        Handle exceptions by logging error information and returning a user message.
        A static method which captures exceptions, logs them, and creates responses to guide users
        through resolving the problem. It generates a list of messages containing both an assistant and
        a user message providing actionable error information.
        
        Args:
            e (Exception):
                 The exception that was raised.
            assistant_message (str):
                 A predefined message intended for the assistant component.
            context (ContextBase):
                 The context in which the error occurred; includes a logger.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage objects containing the assistant message and
                the error information directed to the user.
            

        """
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _parse_response(
        self, context: ContextBase, response: str, chain_results: List[ChatMessage]
    ) -> List[ScoredChatMessage]:
        """
        Parses the response from a list of chat messages associated with grades and returns a list of scored chat messages.
        This function processes a raw string response containing potential grades for chat messages. It attempts to parse
        individual grades from the response and associate them with their corresponding chat message in the
        `chain_results` list. If a grade for a message cannot be parsed, it raises a `LLMParsingException`. If there
        are any missing grades after parsing, it raises an `EvaluatorException` indicating which message indexes
        are missing grades.
        
        Args:
            context (ContextBase):
                 The context in which the function is called, used for logging purposes.
            response (str):
                 The raw string response containing the lines of grades to be parsed.
            chain_results (List[ChatMessage]):
                 A list of chat messages that are expected to be associated with grades.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of scored chat messages, which include the original chat message and the
                assigned grade.
        
        Raises:
            LLMParsingException:
                 If none of the grades could be parsed following the formatting instructions.
            EvaluatorException:
                 If there are missing grades for one or more chat messages after parsing is complete.

        """
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        grades = [r.unwrap() for r in parsed if r.is_some()]
        if len(grades) == 0:
            raise LLMParsingException("None of your grade could be parsed. Follow exactly formatting instructions.")

        scored_messages = []
        missing_indexes = []
        for idx, message in enumerate(chain_results):
            try:
                grade = next(filter(lambda item: item.index == (idx + 1), grades))
                scored_message = ScoredChatMessage(
                    ChatMessage.agent(message=message.message, data=message.data), grade.grade
                )
                scored_messages.append(scored_message)
                context.logger.debug(f"{grade} Graded message: `{message.message}`")
            except StopIteration:
                missing_indexes.append(idx + 1)

        if len(missing_indexes) > 1:
            missing_msg = f"Missing grade for the answers with indexes {missing_indexes}."
            raise EvaluatorException(f"Grade ALL {len(chain_results)} answers. {missing_msg}")

        if len(missing_indexes) > 0:
            missing_msg = f"Missing grade for the answer with index {missing_indexes[0]}."
            raise EvaluatorException(f"Grade ALL {len(chain_results)} answers. {missing_msg}")

        return scored_messages

    def _build_llm_messages(self, query: ChatMessage, skill_messages: List[ChatMessage]) -> List[LLMMessage]:
        """
        Builds a list of LLMMessage instances from a user query and a list of skill chat messages.
        This method takes a user's chat message and a list of messages from various skills, uses them to
        create system and user messages, and encapsulates them into LLMMessage instances. The method returns
        a list containing at least one system message followed by a user message constructed from the input data.
        
        Args:
            query (ChatMessage):
                 The chat message sent by the user.
            skill_messages (List[ChatMessage]):
                 A list of messages derived from different skills responding to the user's query.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage instances that includes a system message and a user message based
                on the provided chat messages from the skills. Returns an empty list if there are no messages from the skills.
            

        """
        if len(skill_messages) <= 0:
            return []

        responses = [skill_message.message for skill_message in skill_messages]
        return [self._build_system_message(), self._build_user_message(query.message, responses)]

    def _parse_line(self, line: str) -> Option[SpecialistGrade]:
        """
        Parses a given line of text to produce a `SpecialistGrade` object if possible. The line is expected to contain a field separator, as defined by the `LLMAnswer.field_separator()` method. If the separator is not found within the line, the method returns an `Option.none()`, indicating that the parsing failed and no `SpecialistGrade` object can be constructed from the line provided. If the separator is found, the method attempts to create a `SpecialistGrade` object through the `_llm_answer.to_object(line)` call. The result, which could be `None` or a valid `SpecialistGrade`, is wrapped into an `Option` and returned.
        
        Args:
            line (str):
                 The text line to parse.
        
        Returns:
            (Option[SpecialistGrade]):
                 An `Option` object encapsulating the `SpecialistGrade`, or none if parsing fails.

        """
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[SpecialistGrade] = self._llm_answer.to_object(line)
        return Option(cs)

    @staticmethod
    def _build_user_message(query: str, answers: list[str]) -> LLMMessage:
        """
        Builds a user message for grading based on a question and a list of answers.
        This static method constructs a message that includes a given question followed by a list of answers. Each answer is enumerated and formatted for the user to grade according to specific instructions. If an answer is empty, the word 'EMPTY' is displayed instead.
        
        Args:
            query (str):
                 The question for which answers are being graded.
            answers (list[str]):
                 A list of answers to be graded, corresponding to the question.
        
        Returns:
            (LLMMessage):
                 A message object containing the formatted prompt for the user to grade the answers.

        """
        prompt_answers = "\n".join(
            f"- answer #{index + 1} is: {answer if len(answer) > 0 else 'EMPTY'}"
            for index, answer in enumerate(answers)
        )
        lines = [
            "The question to grade is:",
            query,
            "Please grade the following answers according to your instructions:",
            prompt_answers,
        ]
        prompt = "\n".join(lines)
        return LLMMessage.user_message(prompt)

    def _build_system_message(self) -> LLMMessage:
        """
        Constructs a system message encapsulating instructions and role description for grading purposes.
        This method assembles a comprehensive message intended to guide an instructor who evaluates answers provided by different Specialists. It outlines the grading criteria, emphasizing objectivity and content accuracy, and provides a formatting guide for both the questions and the responses.
        
        Returns:
            (LLMMessage):
                 An instance of LLMMessage containing the formatted system prompt.
            

        """
        task_description = [
            "\n# ROLE",
            "You are an instructor, with a large breadth of knowledge.",
            "You are grading with objectivity answers from different Specialists to a given question.",
            "\n# INSTRUCTIONS",
            "1. Give a grade from 0.0 to 10.0",
            "2. Evaluate carefully the question and the proposed answer.",
            "3. Ignore how assertive the answer is, only content accuracy count for grading."
            "4. Consider only the Specialist's answer and ignore its index for grading.",
            "5. Ensure to be consistent in grading, identical answers must have the same grade.",
            "6. Irrelevant, inaccurate, inappropriate, false or empty answer must be graded 0.0",
            "\n# FORMATTING",
            "1. The list of given answers is formatted precisely as:",
            "- answer #{index} is: {Specialist's answer or EMPTY if no answer}",
            "2. For each given answer, format your response precisely as:",
            self._llm_answer.to_prompt(),
        ]
        prompt = "\n".join(task_description)
        return LLMMessage.system_message(prompt)
