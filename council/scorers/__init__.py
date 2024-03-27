"""

Module __init__ for the scorer package.

This module serves as an initializer for the scorer package, making core components
available for use when imported. It consolidates the essential classes needed to
handle scoring logic based on different criteria, and also define custom exceptions
related to this scoring process.

Classes:
    ScorerException -- A custom exception class for scorer-specific errors.
    ScorerBase -- An abstract base class that defines the interface and shared logic
                  for scorers.
    LLMSimilarityScorer -- A concrete implementation of ScorerBase that assesses
                          similarity between messages using a Large Language
                          Model (LLM).


"""
from .scorer_exception import ScorerException
from .scorer_base import ScorerBase
from .llm_similarity_scorer import LLMSimilarityScorer
