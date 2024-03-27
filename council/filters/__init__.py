"""

Module __init__

This module initializes the filtering system by importing the core components of the filter architecture.
The filter architecture includes the central exception class for filtering-related issues, the abstract base class for filter implementations,
and concrete implementations of filters for specific tasks.

Imports:
    FilterException: A custom exception class used for handling exceptions related to the filtering process.
    FilterBase: An abstract base class that defines the interface and common behaviour for filters.
    BasicFilter: A concrete implementation of FilterBase that filters based on score thresholds and top-k selection.
    LLMFilter: A concrete implementation that uses a Language Model (LLM) to perform more sophisticated filtering.



"""

from .filter_base import FilterException, FilterBase
from .basic_filter import BasicFilter
from .llm_filter import LLMFilter
