"""

A module that provides a controller to integrate language models (LLMs) into a system for handling specialist evaluations and responses based on user input. The module defines specialized classes and exceptions to instantiate and control the behavior of LLMs within an evaluation context, ensuring that the LLM can interpret instructions, score specialists, and validate responses accordingly. The main controller class, LLMController, extends ControllerBase and manages interactions with the MonitoredLLM component and defines how to execute sequences of instructions, parse responses, handle errors, and retry logic. Additionally, utility classes like Specialist and custom exceptions such as LLMParsingException provide structured way to represent specialist data and error handling for the parsing process. The module relies heavily on decorators such as llm_property and llm_class_validator to enrich the specialist model with behavior suitable for interaction with LLMs.


"""
from typing import List, Optional, Sequence, Tuple
from typing_extensions import TypeGuard

from council.chains import ChainBase
from council.contexts import AgentContext, ChatMessage, ContextBase
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.utils import Option
from council.controllers import ControllerBase, ControllerException
from .execution_unit import ExecutionUnit
from council.llm.llm_answer import llm_property, LLMAnswer, LLMParsingException, llm_class_validator


class Specialist:
    """
    A class representing a specialist with a scoring system and associated details.
    
    Attributes:
        _name (str):
             The name of the specialist.
        _justification (str):
             The justification for the specialist's score.
        _instructions (str):
             Instructions associated with the specialist.
        _score (int):
             The score awarded to the specialist, expected to be between 0 and 10.
    
    Methods:
        __init__:
             Initializes a new instance of the Specialist class.
        name:
             Property that gets the specialist's name.
        score:
             Property that gets the specialist's score.
        instructions:
             Property that gets the instructions associated with the specialist.
        justification:
             Property that gets the justification for the specialist's score.
        __str__:
             Returns a formatted string representation of the specialist.
        validate:
             Validates whether the specialist's score is within the valid range.
    
    Raises:
        LLMParsingException:
             If the score is not within the range of 0 to 10.

    """
    def __init__(self, name: str, justification: str, instructions: str, score: int):
        """
        Initializes a new instance of the class with provided name, justification, instructions, and score.
        
        Args:
            name (str):
                 A string representing the name attribute of the instance.
            justification (str):
                 A string providing justification for creating the instance.
            instructions (str):
                 Detailed instructions associated with the instance.
            score (int):
                 An integer representing the score attribute of the instance.

        """
        self._instructions = instructions
        self._score = score
        self._name = name
        self._justification = justification

    @llm_property
    def name(self) -> str:
        """
        Gets the name of the object.
        
        Returns:
            (str):
                 The current value of the private `_name` attribute.

        """
        return self._name

    @llm_property
    def score(self) -> int:
        """
        
        Returns the score associated with an instance.
            This method retrieves the private _score attribute from the instance it is called on and returns it as an integer, which represents the current score. This is a property method, intent is to provide read-only access to the _score attribute.
        
        Returns:
            (int):
                 The current score of the instance.

        """
        return self._score

    @llm_property
    def instructions(self) -> str:
        """
        Retrieves the stored set of instructions.
        This method acts as a getter to retrieve the private `_instructions` attribute that contains
        a set of instructions.
        
        Returns:
            (str):
                 The current set of instructions stored within the instance.
            

        """
        return self._instructions

    @llm_property
    def justification(self) -> str:
        """
        
        Returns the justification of an object.
            This property-based accessor method returns the justification information
            associated with an object. The justification encapsulates the rationale or
            reasoning behind the object’s creation, existence, or state.
        
        Returns:
            (str):
                 The justification information for the object.

        """
        return self._justification

    def __str__(self):
        """
        Convert the specialist object to a human-readable string representation.
        Overrides the default `__str__` method to provide a custom string representation
        of the specialist object. It includes the specialist's name, score, and justification for the
        score in a formatted string.
        
        Returns:
            (str):
                 A string representation of the specialist object with name, score, and justification.
            

        """
        return (
            f"The specialist `{self._name}` was scored `{self._score}` with the justification `{self._justification}`"
        )

    @llm_class_validator
    def validate(self):
        """
        Validates the specialist's score attribute.
        This method checks if the specialist's score is within the acceptable range between 0 and 10 (inclusive). If the score is outside of this range, it raises an LLMParsingException with an error message indicating that the score is invalid.
        
        Raises:
            LLMParsingException:
                 If the `_score` attribute is less than 0 or greater than 10.
            

        """
        if self._score < 0 or self._score > 10:
            raise LLMParsingException(f"Specialist's score `{self._score}` is invalid, value must be between 0 and 10.")


class LLMController(ControllerBase):
    """
    A controller that orchestrates Long-Lifetime Models (LLMs) to interact with a set of chains.
    The LLMController is responsible for parsing responses from the LLM, handling errors, and
    constructing execution plans based on the relevance scores of specialists. It leverages monitored
    instances of LLMs to post chat requests and uses internal mechanisms to retry on parsing failures.
    This controller is initialized with a list of chains, an LLM base, a response threshold, an optional
    parameter for the number of top-k relevant specialists, and a flag for parallelism.
    
    Attributes:
        _llm (MonitoredLLM):
             The monitored LLM instance used to post chat requests.
        _response_threshold (float):
             A threshold value above which the specialists are considered relevant.
        _top_k (int):
             The number of top relevant specialists to be considered for execution.
        _llm_answer (LLMAnswer):
             Represents the expected answer format from the LLM.
        _llm_system_message (LLMMessage):
             The system message to be used by the LLM for context.
        _retry (int):
             The number of retries allowed for executing LLM consultation in case of parsing failures.
    
    Raises:
        Exception:
             If no user message is found or the necessary parsing requirements are not met.
        ControllerException:
             If the execution plan cannot be created or required scores for chains are missing.
        LLMParsingException:
             If the response from the LLM could not be parsed.

    """

    _llm: MonitoredLLM

    def __init__(
        self,
        chains: Sequence[ChainBase],
        llm: LLMBase,
        response_threshold: float = 0.0,
        top_k: Optional[int] = None,
        parallelism: bool = False,
    ):
        """
        Initializes a new instance of the class, setting up its dependencies and configuration.
        This constructor instantiates the class with a specific set of chains to be monitored, a language model, and various parameters to control behavior. It leverages the base class initialization logic to handle parallelism if required and registers the language model as a monitored component. The response threshold and top-k chains are also configured, along with the instantiation of an LLMAnswer object which is used for parsing responses from the language model.
        
        Args:
            chains (Sequence[ChainBase]):
                 A sequence of ChainBase instances that the class will monitor.
            llm (LLMBase):
                 An instance of LLMBase representing the language model to be used.
            response_threshold (float, optional):
                 The minimum score threshold for responses to be considered valid. Defaults to 0.0.
            top_k (Optional[int], optional):
                 The number of top-scoring chains to consider in the evaluation. If None, all chains are considered. Defaults to None.
            parallelism (bool, optional):
                 A flag indicating whether to run the monitoring operations in parallel across the chains. Defaults to False.
        
        Raises:
            ValueError:
                 If the provided top_k value is negative or if other configuration values are invalid.

        """
        super().__init__(chains=chains, parallelism=parallelism)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._response_threshold = response_threshold
        if top_k is None:
            self._top_k = len(self._chains)
        else:
            self._top_k = min(top_k, len(self._chains))
        self._llm_answer = LLMAnswer(Specialist)
        self._llm_system_message = self._build_system_message()
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        """
        Execute the series of actions based on an agent's context and LLM messages.
        This method orchestrates the process of constructing messages for the LLM, submitting them, and parsing the LLM responses to generate a list of execution units with their respective ranks, sorted by the rank in descending order. A retry mechanism is built-in, allowing for multiple attempts if parsing fails. The result is trimmed to include only the top_k execution units that meet the specified response threshold.
        
        Args:
            context (AgentContext):
                 The context object providing relevant information for execution, which includes configuration and state required for making the LLM request.
        
        Returns:
            (List[ExecutionUnit]):
                 A list of execution units sorted by their ranks in descending order. Only the top-ranked units that meet the response threshold are included, up to a maximum of top_k units.
        
        Raises:
            LLMParsingException:
                 If the response from the LLM cannot be parsed successfully after the designated number of retries.
            ControllerException:
                 If the controller encounters an exception during the execution which is not resolved after retries.
            

        """
        retry = self._retry
        messages = self._build_llm_messages(context)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                retry -= 1
                plan = self._parse_response(context, response)
                plan.sort(key=lambda item: item[1], reverse=True)
                return [item[0] for item in plan if item[1] >= self._response_threshold][: self._top_k]
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except ControllerException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise ControllerException("LLMController failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        """
        Handle an occurred exception, log it, and prepare assistant messages for the user.
        This static method is designed to handle exceptions that occur within the logic of an application,
        log an appropriate warning message including the exception details, and then create error
        messages that guide the user based on the assistant's message.
        
        Args:
            e (Exception):
                 The exception that was caught and needs to be handled.
            assistant_message (str):
                 A predefined message from the assistant to help guide the user
                after the error occurred.
            context (ContextBase):
                 The context in which the exception occurred. This typically
                includes the execution environment and logging facilities.
        
        Returns:
            (List[LLMMessage]):
                 A list containing two LLMMessage objects;
                one with the assistant message to guide the user,
                and the other with the error details that inform the user what went wrong.
            

        """
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _build_llm_messages(self, context: AgentContext) -> List[LLMMessage]:
        """
        Constructs a list of LLMMessage objects for a specific agent context.
        This method takes the context of an agent and builds the necessary Large Language Model (LLM) messages to be processed,
        which includes a system message pre-defined within the class instance and a user message that is generated based on the context provided.
        
        Args:
            context (AgentContext):
                 The context of the agent for which the messages need to be created. It contains all the necessary information
                that will be used to build the user-specific message.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage objects. The first message is always the system-level message followed by the user-specific
                message crafted based on the provided context.
            

        """
        return [self._llm_system_message, self._build_user_message(context)]

    def _build_user_message(self, context: AgentContext) -> LLMMessage:
        """
        Constructs a user message for the language model based on the current context.
        This function builds a message by concatenating information about available
        chains with the main instruction for the language model. It appends this information
        to the last user message found in the chat history of the given context. If no
        user message is present in the chat history, the function raises an exception.
        
        Parameters:
            context (AgentContext):
                 The context object that contains the chat history and
                other relevant information needed to build the message.
        
        Returns:
            (LLMMessage):
                 An instance of LLMMessage representing the user message constructed
                for the language model.
        
        Raises:
            Exception:
                 If the chat history does not contain any user messages.

        """
        message = context.chat_history.try_last_user_message
        if message.is_none():
            raise Exception("No user message.")

        user_message = "\n".join(
            ["# SPECIALISTS"]
            + [f"name: {c.name};description: {c.description};{c.is_supporting_instructions}" for c in self._chains]
            + [f"\n{self._get_main_instruction()} for:\n `{message.unwrap().message}`"]
        )
        return LLMMessage.user_message(user_message)

    def _build_system_message(self) -> LLMMessage:
        """
        Generates a system message encapsulating the task's role, instructions, and formatting guidelines for scoring the relevance of Specialists.
        This method constructs a message tailored for a system entity responsible for rating Specialists according to their relevance to a user task. The relevance is scored on a scale from 0 (irrelevant) to 10 (highly relevant). The message is composed of clear role definition, scoring instructions, and precise formatting rules for both the Specialist list and the expected response format. It collects the main instruction using an internal method, compiles the task description and instructions into a structured format, and finally converts the aggregated content into an LLMMessage object ready for dispatch.
        
        Returns:
            (LLMMessage):
                 A system message object containing formatted instructions and guidelines for scoring Specialist relevance.

        """
        instruction = self._get_main_instruction()
        task_description = [
            "# ROLE",
            "You are a knowledgeable expert responsible to fairly score Specialists.",
            "The score will reflect how relevant is a Specialist to solve or execute a user task.",
            "\n# INSTRUCTIONS",
            f"1. {instruction}.",
            "2. Read carefully the user task and the Specialist description to score its relevance.",
            "3. Score from 0 (poor relevance or out of scope) to 10 (perfectly relevant).",
            "4. Ignore Specialist's name or its order in the list to give your score.",
            "5. If Specialist is supporting instructions, give any useful instructions to execute the user task.",
            "\n# FORMATTING",
            "1. Specialist list is precisely formatted as:",
            "name: {name};description: {description};{boolean indicating if Specialist is supporting instructions}",
            "2. Your response is precisely formatted as:",
            self._llm_answer.to_prompt(),
        ]
        return LLMMessage.system_message("\n".join(task_description))

    def _get_main_instruction(self):
        """
        Gets the main instruction for scoring based on the '_top_k' attribute value.
        This private method determines what instruction to return based on the value of the '_top_k' attribute which is expected to be a part of the instance state. If '_top_k' is set to 1, it implies that only the single most relevant and best Specialist should be scored. Otherwise, it implies that all Specialists are to be scored without that restriction.
        
        Returns:
            (str):
                 A string containing the relevant instruction for scoring Specialists.

        """
        if self._top_k == 1:
            return "Score only the most relevant and best Specialist"
        return "Score all Specialists"

    def _parse_response(self, context: AgentContext, response: str) -> List[Tuple[ExecutionUnit, int]]:
        """
        Parse the response from the agent context and returns a list of tuples with execution units and their scores.
        This method takes the raw response string from an agent, splits it into lines, parses each line,
        and filters out any unparsable or empty results. It also manages the scoring based on the top_k
        attribute. If all lines are unparsable, it raises an LLMParsingException. Moreover, if the
        required chains are missing in a multi-chain scenario (self._top_k > 1) or if multiple specialists
        are scored when only the most relevant should be (self._top_k == 1), a ControllerException is raised.
        
        Args:
            context (AgentContext):
                 The context of the agent providing settings and state relevant for parsing.
            response (str):
                 The raw response string received from an agent.
        
        Returns:
            (List[Tuple[ExecutionUnit, int]]):
                 A list of tuples where each tuple contains an ExecutionUnit and
                its corresponding score (int).
        
        Raises:
            LLMParsingException:
                 If none of the response lines can be parsed as per the formatting instructions.
            ControllerException:
                 If there is a missing score for any required chain or if multiple specialists
                are scored when only one should be.
            

        """
        parsed = [self._parse_line(context, line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        if len(filtered) == 0:
            raise LLMParsingException("None of your response could be parsed. Follow exactly formatting instructions.")

        if self._top_k > 1:
            actual_chains = [item[0].chain.name for item in filtered]
            missing_chains = [chain.name for chain in self._chains if chain.name not in actual_chains]
            if len(missing_chains) > 0:
                raise ControllerException(f"Missing scores for {missing_chains}. Follow exactly your instructions.")

        if len(filtered) != 1 and self._top_k == 1:
            raise ControllerException("You scored multiple Specialists. Score ONLY the most relevant specialist.")

        return filtered

    def _parse_line(self, context: AgentContext, line: str) -> Option[Tuple[ExecutionUnit, int]]:
        """
        Parses a line to extract an execution unit and its associated score, if applicable.
        This method takes a line of text and attempts to interpret it as containing information
        about a Specialist's performance and score. It processes the line and, if successful, returns an
        option containing a tuple with a created Execution Unit and its score; otherwise, it returns
        an option with 'None'.
        
        Args:
            context (AgentContext):
                 The context in which the agent is operating, used here for logging.
            line (str):
                 The line of text to be parsed which may contain scored specialist information.
        
        Returns:
            (Option[Tuple[ExecutionUnit, int]]):
                 An option instance that contains a tuple of an
                ExecutionUnit and an integer representing its score if the line could be parsed;
                otherwise 'None' if the line does not contain the appropriate information or
                a chain matching the specialist's name could not be found.
        
        Raises:
            ControllerException:
                 If a chain with a name matching that of the specialist is
                not found within the available chains, indicating a data mismatch or configuration error.

        """
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        scored_specialist: Optional[Specialist] = self._llm_answer.to_object(line)
        if scored_specialist is not None:

            def typeguard_predicate(chain_base: ChainBase) -> TypeGuard[ChainBase]:
                """
                Checks if the given 'chain_base' object is an instance of ChainBase and its name matches
                the case-folded 'scored_specialist' name.
                
                Args:
                    chain_base (ChainBase):
                         An object to test if it is of the ChainBase type and
                        if its name attribute matches the scored_specialist's name.
                
                Returns:
                    (TypeGuard[ChainBase]):
                         A boolean value representing whether the 'chain_base' is
                        an instance of ChainBase with a matching name.
                    

                """
                return (
                    isinstance(chain_base, ChainBase)
                    and chain_base.name.casefold() == scored_specialist.name.casefold()
                )

            try:
                chain = next(filter(typeguard_predicate, self._chains))
                context.logger.debug(f"{scored_specialist}")
                return Option.some(
                    (
                        self._build_execution_unit(
                            chain, context, scored_specialist.instructions, scored_specialist.score
                        ),
                        scored_specialist.score,
                    )
                )
            except StopIteration:
                context.logger.warning(f'message="no chain found with name `{scored_specialist.name}`"')
                raise ControllerException(f"The Specialist `{scored_specialist.name}` does not exist.")
        return Option.none()

    def _build_execution_unit(
        self, chain: ChainBase, context: AgentContext, instructions: str, score: int
    ) -> ExecutionUnit:
        """
        Build and return an `ExecutionUnit` based on the provided chain and context.
        This function takes a `ChainBase` instance, an `AgentContext`, a string of instructions, and a score integer to prepare
        an `ExecutionUnit`. It initializes an `ExecutionUnit` with the given chain and the budget from the context. Optionally,
        it creates an initial chat message state if the chain supports instructions. The name of the execution unit is set using the
        chain's name combined with the score, and the default rank is assigned.
        Arguments:
        chain (ChainBase): The chain object to associate with the new execution unit.
        context (AgentContext): The context providing the budget for the execution unit.
        instructions (str): Instruction message to potentially initialize the `ExecutionUnit`. It will be used to create an
        initial state for the execution unit if the chain supports instructions.
        score (int): An integer score that is appended to the chain's name to form the execution unit's name.
        
        Returns:
            (ExecutionUnit):
                 A newly created `ExecutionUnit` instance with the provided chain, context's budget, and optionally
                with an initial state and a customized name composed of the chain's name and the score.

        """
        return ExecutionUnit(
            chain,
            context.budget,
            initial_state=ChatMessage.chain(message=instructions) if chain.is_supporting_instructions else None,
            name=f"{chain.name};{score}",
            rank=self.default_execution_unit_rank,
        )
