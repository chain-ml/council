"""

This module initializes the package by importing the PromptBuilder class.

The PromptBuilder class is designed to facilitate the creation of custom prompts based on certain templates
and a collection of instructions or other contextual data from a chat history. It allows applying dynamic content
into a fixed template, providing an efficient way to generate formatted strings for prompts.

Classes:
    PromptBuilder: A class used to construct prompts dynamically by applying contextual data to
                   a predefined template structure.


"""
from .prompt_builder import PromptBuilder
