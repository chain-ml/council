"""

A module to calculate and score similarity between messages using a language representation model (LLM).

This module defines classes associated with calculating the similarity score between an expected
message and an actual message in a chat-based system. It uses a large language model (LLM)
to perform this calculation and validate the format of the response with regard to the similarity score.
The main components of the module can be broken down into the following classes:

- `SimilarityScore`:
  Represents the similarity score with an associated justification string.
  It provides properties to access the normalized score and justification.
  It also includes validation to ensure scores fall within a valid range.

- `LLMSimilarityScorer`:
  Inherits from `ScorerBase` and provides functionality to calculate a similarity score
  between an expected and an actual message using an LLM.
  It handles retries and errors during the process of scoring.
  Additionally, it includes a mechanism to parse the LLM's response into a `SimilarityScore` object.

Classes:
- SimilarityScore
- LLMSimilarityScorer

Exceptions:
- LLMParsingException

Validators:
- llm_class_validator

Dependencies:
- ScorerException
- ScorerBase
- ChatMessage
- ScorerContext
- ContextBase
- LLMBase
- LLMMessage
- MonitoredLLM
- llm_property
- LLMAnswer
- LLMParsingException
- llm_class_validator
- Option

The module ensures integration of the scoring functionality within a larger system to monitor and score real-time messages by comparing them against an expected message pattern or standard.


"""
from typing import List, Dict, Any, Optional

from . import ScorerException
from .scorer_base import ScorerBase
from council.contexts import ChatMessage, ScorerContext, ContextBase
from council.llm import LLMBase, LLMMessage, MonitoredLLM, llm_property, LLMAnswer
from ..llm.llm_answer import LLMParsingException, llm_class_validator
from ..utils import Option


class SimilarityScore:
    """
    A class that encapsulates a similarity score along with its justification.
    
    Attributes:
        _score (float):
             A private float value representing the similarity score, from 0 to 100.
        _justification (str):
             A private string providing the reasoning or justification for the given score.
    
    Methods:
        __init__(self, score:
             float, justification: str):
            Initializes a new instance of SimilarityScore with a score and its justification.
        score(self) -> float:
            A property that returns the normalized score as a float between 0.0 and 1.0.
        justification(self) -> str:
            A property that returns the justification for the similarity score.
        __str__(self):
            Returns a string representation of the similarity score and its justification.
        validate(self):
            Validates the similarity score ensuring it is within the acceptable range of 0 to 100. If not,
            raises an LLMParsingException with an error message.

    """
    def __init__(self, score: float, justification: str):
        """
        Initializes a new instance of the class with a score and its justification.
        
        Args:
            score (float):
                 The numerical assessment score.
            justification (str):
                 A textual explanation of the given score.

        """
        self._score = score
        self._justification = justification

    @llm_property
    def score(self) -> float:
        """
        Computes the adjusted score value as a float by dividing the internal score by 100.0.

        """
        return self._score / 100.0

    @llm_property
    def justification(self) -> str:
        """
        Retrieves the justification attribute of the object.
        This property method returns the justification information stored within an object. It is intended to provide access to an internal '_justification' attribute, which should hold a string that describes the justification of the object in some context.
        
        Returns:
            (str):
                 The justification as a string.

        """
        return self._justification

    def __str__(self):
        """
        
        Returns a human-readable string representation of the object, including the similarity score and its justification.
        
        Returns:
            (str):
                 A string that describes the similarity score and the justification for that score.

        """
        return f"Similarity score is {self.score} with the justification: {self._justification}"

    @llm_class_validator
    def validate(self):
        """
        Validates that the similarity score is within the acceptable range.
        This method checks if the object's score is between 0 and 100. If the
        score is outside of this range, it raises an LLMParsingException with a
        message indicating the invalid score value.
        
        Raises:
            LLMParsingException:
                 If the `_score` attribute is less than 0 or
                greater than 100, indicating an invalid value.

        """
        if self._score < 0 or self._score > 100:
            raise LLMParsingException(f"Similarity Score `{self._score}` is invalid, value must be between 0 and 100.")


class LLMSimilarityScorer(ScorerBase):
    """
    Class that computes the similarity score between an expected message and an actual message using a language model.
    This class extends the ScorerBase class and implements functionality to calculate how similar two text messages are.
    The similarity calculation is performed by a language model, and the result is returned as a float score.
    The class provides methods to convert the score to a dictionary, handle errors during similarity scoring,
    build messages for the language model, and parse the language model's response.
    
    Attributes:
        _llm (MonitoredLLM):
             The language model which computes the similarity score.
        _expected (str):
             The expected text message to which the actual message is compared.
        _llm_answer (LLMAnswer):
             Helper class to format the language model answer adhering to a similarity score response.
        _system_message (LLMMessage):
             Pre-generated system message that contains instructions for the language model.
        _retry (int):
             Number of retries if the language model's response is inadequate or an exception occurs.
    
    Methods:
        to_dict:
             Converts the attributes of this scorer into a dictionary format.
        _score:
             Calculates the similarity score using messages and a given context; handles errors and retries as needed.
        _handle_error:
             Handles either LLMParsingException or ScorerException, logs the warning, and prepares recovery messages.
        _build_llm_messages:
             Constructs messages for the language model based on user and expected messages.
        _build_system_message:
             Creates a system level message to provide context and instructions to the language model.
        _parse_response:
             Parses through the language model's response to extract a similarity score.
        _parse_line:
             Extracts a similarity score from a single line of text provided by the language model.

    """

    def __init__(self, llm: LLMBase, expected: str):
        """
        Initializes a new instance of the class with the provided LLM instance and expected answer string.
        
        Args:
            llm (LLMBase):
                 The instance of LLMBase to be monitored.
            expected (str):
                 The expected answer to be matched with the LLM's output.
        
        Attributes:
            _llm (MonitoredLLM):
                 An LLM instance wrapped in a MonitoredLLM for monitoring purposes.
            _expected (str):
                 Stores the expected answer provided during initialization.
            _llm_answer (LLMAnswer):
                 An instance of LLMAnswer to handle processing and validation of answers.
            _system_message (str):
                 A system message built using the _build_system_message method.
            _retry (int):
                 The number of retries allowed, initialized to 3.

        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._expected = expected
        self._llm_answer = LLMAnswer(SimilarityScore)
        self._system_message = self._build_system_message()
        self._retry = 3

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object's properties to a dictionary format, including the expected property.
        This method extends the `to_dict` method from the superclass, adding the 'expected' property
        from the current instance to the dictionary returned by the superclass method call.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing key-value pairs of the object properties, which includes
                both the base class properties and the 'expected' property of this instance.
            

        """
        result = super().to_dict()
        result["expected"] = self._expected
        return result

    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        Calculate the similarity score based on a message using a language model.
        This function takes a context and a chat message as inputs, builds low-level messages (LLMMessages), and
        communicates with a language model to retrieve a score. If the language model's response is not
        properly formatted or an exception is raised, the function will handle the error and retry the score
        calculation a certain number of times. If all retries fail, the function raises a ScorerException.
        
        Args:
            context (ScorerContext):
                 The context in which the scorer operates, including configurations,
                logging, and other necessary data.
            message (ChatMessage):
                 The message object that needs to be scored.
        
        Returns:
            (float):
                 The similarity score as a floating-point number if the calculation succeeds.
        
        Raises:
            LLMParsingException:
                 An exception is raised when the response from the language model is
                not correctly formatted.
            ScorerException:
                 An exception is raised if the scoring process fails after all retries or if
                there is an error while trying to get the score from the response.
            

        """
        retry = self._retry
        messages = self._build_llm_messages(message)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                retry -= 1
                similarity_score = self._parse_response(context, response)
                return similarity_score.score
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except ScorerException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise ScorerException("LLMSimilarityScorer failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        """
        Handle an exception by logging it and creating LLMMessage instances for feedback.
        This method logs an exception and constructs a list of ``LLMMessage`` objects that describe
        the error to the assistant and user, respectively.
        
        Args:
            e (Exception):
                 The exception that was caught.
            assistant_message (str):
                 A predefined message explaining the error to the assistant.
            context (ContextBase):
                 The context in which the error occurred. Provides access to the logger.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage instances. The first message is intended for
                the assistant to provide a predefined explanation of what went wrong, and the second
                message gives the user a directive to fix the problem, including the exception details.
            

        """
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _build_llm_messages(self, message: ChatMessage) -> List[LLMMessage]:
        """
        Generates a list of LLMMessage instances based on the user prompt and the provided ChatMessage.
        The method constructs a user prompt by asking for the similarity score between an actual message and an expected message. It then creates a sequence of LLMMessage instances, with the first message being a system-generated message, followed by a user message containing the generated prompt.
        
        Args:
            message (ChatMessage):
                 The chat message containing the actual message to be compared with the expected one.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage instances. The first is a system message, followed by a user message comprising the similarity score prompt.

        """
        user_prompt = [
            "Please give the similarity score of the actual message compared to the expected one.",
            "Actual message:",
            message.message,
            "Expected message:",
            self._expected,
        ]

        result = [self._system_message, LLMMessage.user_message("\n".join(user_prompt))]
        return result

    def _build_system_message(self) -> LLMMessage:
        """
        Generates a system message that instructs an expert how to evaluate the similarity between an expected message and an actual message.
        This method constructs a system prompt consisting of a role description, specific comparison instructions, a scoring guide, and a formatting section derived from the `_llm_answer` attribute's prompt content. The prompt is then used to create an `LLMMessage` object representing the system message to be used for guiding the expert.
        
        Returns:
            (LLMMessage):
                 An object representing the structured system message composed of the role description, instructions, scoring guide, and formatting instructions.

        """
        system_prompt = [
            "# ROLE",
            "You are an expert specialized in evaluating how similar an expected message and an actual message are.",
            "\n# INSTRUCTIONS",
            "1. Compare the {expected} message and the {actual} message.",
            "2. Score 0 (2 messages are unrelated) to 100 (the 2 messages have the same content).",
            "3. Your score must be fair.",
            "\n# FORMATTING",
            self._llm_answer.to_prompt(),
        ]
        return LLMMessage.system_message("\n".join(system_prompt))

    def _parse_response(self, context: ContextBase, response: str) -> SimilarityScore:
        """
        def _parse_response(self, context: ContextBase, response: str) -> SimilarityScore:
        
        Parses the response from a context-based request and returns the similarity score.
        The method splits the response string into lines, attempting to parse each line using an internal parsing method. The parsed results are expected to be of an optional type, where only some values contain usable data. It collects all successfully parsed lines, unwraps them to obtain actual values, and filters out any None results. If all lines in the response are not parseable, it raises an exception to indicate failure in parsing.
        The method assumes that the first parsed and successfully unwrapped line contains the similarity score of interest and returns it. Also, logs the similarity score for debugging purposes.
        
        Args:
            context (ContextBase):
                 The context within which this parsing occurs, typically containing logging or other environmental information.
            response (str):
                 The raw string response received from a request that needs to be parsed.
        
        Returns:
            (SimilarityScore):
                 The first successfully parsed similarity score from the response.
        
        Raises:
            LLMParsingException:
                 If none of the lines in the response could be parsed according to expected formatting instructions.

        """
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        if len(filtered) == 0:
            raise LLMParsingException("None of your response could be parsed. Follow exactly formatting instructions.")

        similarity_score = filtered[0]
        context.logger.debug(f"{similarity_score}")
        return similarity_score

    def _parse_line(self, line: str) -> Option[SimilarityScore]:
        """
        Parses a given line of text to extract a `SimilarityScore` object if possible.
        This method attempts to parse a given line of text using a delimiter defined by `LLMAnswer.field_separator()`. If the delimiter is found in the text, it further attempts to create a `SimilarityScore` object from the line using `_llm_answer.to_object(line)`. If the creation is successful, an `Option.some(similarity_score)` is returned, indicating that a valid `SimilarityScore` object has been enclosed in the `Option`. If parsing fails or no delimiter is present, the method returns `Option.none()`, indicating the absence of a `SimilarityScore` object.
        
        Args:
            line (str):
                 The line of text to be parsed.
        
        Returns:
            (Option[SimilarityScore]):
                 An `Option` containing a `SimilarityScore` object if parsing is successful, otherwise `Option.none()`.

        """
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        similarity_score: Optional[SimilarityScore] = self._llm_answer.to_object(line)
        if similarity_score is not None:
            return Option.some(similarity_score)
        return Option.none()
