"""

Module skill_base

This module provides the base class for creating skills within a skills system.
`SkillBase` is an abstract subclass of `SkillRunnerBase`, and it enforces the implementation
of `execute` method, with additional methods for generating success and error messages.

Classes:
    SkillBase(SkillRunnerBase): A base class for skill implementation in the system.

Functions:
    None

Attributes:
    _name (str): The name of the skill.



"""
from __future__ import annotations

from typing import Any
from abc import abstractmethod

from council.contexts import SkillContext, ChatMessage
from council.runners import SkillRunnerBase


class SkillBase(SkillRunnerBase):
    """
    A base class for creating skill-based functionalities within a chat-oriented application.
    This abstract class serves as a foundational component for defining skills, providing common
    methods and attributes that can be utilized or overridden by derived skill classes. It
    implements default behaviors for skill execution management, success and error message generation,
    and standardized logging.
    
    Attributes:
        _name (str):
             The name of the skill, intended to be used primarily internally.
    
    Methods:
        __init__:
            Initializes the SkillBase instance with the provided skill name and sets up the
            base functionality from SkillRunnerBase.
        name:
            A property that gets the skill's name.
        execute:
            An abstract method that must be implemented by derived classes, defining the logic
            for the skill's execution given a SkillContext.
        build_success_message:
            Constructs a success ChatMessage with the given message and optional data, including
            the skill's name as the source.
        build_error_message:
            Constructs an error ChatMessage with the given message and optional data, including
            the skill's name as the source.
        execute_skill:
            Manages the execution process of the skill, logging the start and end of the
            execution, as well as the success or warning as appropriate. It calls the abstract
            execute method and returns its ChatMessage result.
        __repr__:
            Provides the representation of the SkillBase instance, which includes the skill's name.
        __str__:
            Returns a more human-readable string that denotes the Skill instance and its name.

    """

    _name: str

    def __init__(self, name: str):
        """
        Initializes a new instance with the given name.
        
        Args:
            name (str):
                 The name to assign to the instance.
            

        """
        super().__init__(name)
        self._name = name

    @property
    def name(self):
        """
        Gets the name attribute of the class instance.
        This property method is used to retrieve the private _name attribute from an instance of the class.
        It is a 'getter' function that is decorated with @property to make it accessible like an attribute without the need to explicitly call it as a method.
        
        Returns:
            (str):
                 The value of the _name attribute representing the name of the instance.

        """
        return self._name

    @abstractmethod
    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Performs the abstract execution operation utilizing the provided context information.
        This method should be implemented by subclasses to define the specific execution behavior required.
        The implementation is responsible for generating a ChatMessage object based on the given context.
        
        Args:
            context (SkillContext):
                 An object carrying contextual information for the execution of the skill,
                such as user information, conversation history, and any other relevant data.
        
        Returns:
            (ChatMessage):
                 The result of the execution as a ChatMessage object, which should encapsulate
                the response and any associated metadata.
        
        Raises:
            NotImplementedError:
                 If this method is not implemented by a subclass.

        """
        pass

    def build_success_message(self, message: str, data: Any = None) -> ChatMessage:
        """
        Builds a success message encapsulated in a ChatMessage object.
        This function takes a message string and an optional data object, and returns
        a ChatMessage instance representing a non-error message from the current skill.
        
        Args:
            message (str):
                 The main content of the message to be communicated.
            data (Any, optional):
                 Any additional data to be included with the message. Defaults to None.
        
        Returns:
            (ChatMessage):
                 An instance of ChatMessage with the message, data, and source,
                indicating a successful operation or status.
            

        """
        return ChatMessage.skill(message, data, source=self._name, is_error=False)

    def build_error_message(self, message: str, data: Any = None) -> ChatMessage:
        """
        Creates a ChatMessage object containing an error message and associated data.
        This method constructs a ChatMessage object that encapsulates a message string and optional data payload and marks it as an error. The message originates from the skill identified by the object's '_name' attribute.
        
        Args:
            message (str):
                 A descriptive error message string.
            data (Any, optional):
                 Additional data related to the error. This can be of any type. Defaults to None.
        
        Returns:
            (ChatMessage):
                 An instance of ChatMessage with the error information and attributes set.
            

        """
        return ChatMessage.skill(message, data, source=self._name, is_error=True)

    def execute_skill(self, context: SkillContext) -> ChatMessage:
        """
        Gets the default execution unit rank for the current instance.
        This property determines the default rank of the execution unit based on parallelism capability. If parallelism is enabled, it returns 1, otherwise it returns None, indicating there's no default rank specified.
        
        Returns:
            (Optional[int]):
                 The default execution unit rank, 1 if parallelism is enabled, otherwise None.

        """
        context.logger.info(f'message="skill execution started" skill="{self.name}"')
        skill_message = self.execute(context)
        if skill_message.is_ok:
            context.logger.info(
                f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
            )
        else:
            context.logger.warning(
                f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
            )
        return skill_message

    def __repr__(self):
        """
        
        Returns the official string representation of the SkillBase object.
            The `__repr__` method is used to obtain a string representation of the object which is unambiguous and, if possible, matches the expression needed to recreate the object. This representation is mainly used for debugging and development. In this case, it represents the object in the format 'SkillBase(<name>)' where <name> is the name of the skill.
        
        Returns:
            (str):
                 A string that represents the SkillBase object in the format 'SkillBase(<name>)'.

        """
        return f"SkillBase({self.name})"

    def __str__(self):
        """
        
        Returns a string representation of the object with its associated skill name.
            This method overrides the special `__str__` method and is used to create a human-readable string that represents the
            object.
            The string includes the skill name which is an attribute of the object.
        
        Returns:
            (str):
                 A string that represents the object including the word 'Skill' followed by the skill name.

        """
        return f"Skill {self.name}"
