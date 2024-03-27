"""

Module BasicFilter provides a simple filtering mechanism based on score thresholds and max number of items (top-k) for chat messages within an agent context.

This module contains a `BasicFilter` class which extends `FilterBase`. The `BasicFilter` uses a score threshold and/or top-k parameter to filter a list of scored chat messages. The class contains two main private methods, `_execute` which applies filtering based on provided parameters, and `_filter` that actually filters chat messages based on the score threshold.

Classes:
    BasicFilter(FilterBase): A filter class which lets you filter chat messages based on score thresholds and top-k.

Attributes:
    _score_threshold (Optional[float]): The minimum score threshold for a chat message to be included in the filtered result. If None, no minimum score threshold filtering is applied.
    _top_k (Optional[int]): The maximum number of top scoring messages to return. If None or non-positive, no top-k filtering is applied.

Methods:
    __init__(score_threshold: Optional[float], top_k: Optional[int]): Constructs a `BasicFilter` object with optional score threshold and top-k limiting settings.
    _execute(context: AgentContext): Apply the filter on the provided `AgentContext` based on defined score threshold and top-k attributes.
    _filter(context: AgentContext): Perform the actual filtering of chat messages from the `AgentContext` based on the score threshold.


"""
from typing import List, Optional

from council.contexts import AgentContext, ScoredChatMessage
from council.filters import FilterBase


class BasicFilter(FilterBase):
    """
    A class that inherits from FilterBase and provides basic filtering functionality for chat messages.
    BasicFilter allows filtering of chat messages based on score threshold and selecting top ranked messages.
    It initializes with optional parameters to determine the filtering behavior and provides
    methods to execute the filtering process.
    
    Attributes:
        _score_threshold (Optional[float]):
             The minimum score that a chat message must have
            to be passed through the filter. Messages with scores below this threshold will be ignored.
            Defaults to None, which means no filtering is done based on score.
        _top_k (Optional[int]):
             The maximum number of top scoring messages to be returned.
            Only the top k messages as per their scores will be passed through if this parameter is set.
            Defaults to None, which means no limit is applied to the number of messages passed through.
    
    Methods:
        __init__:
             Constructs a new instance of BasicFilter with the given score threshold and top k settings.
        _execute:
             Filters the scored chat messages based on instance settings and returns the list
            of messages that meet the criteria.
        _filter:
             Applies the actual filter logic on the provided chat messages from the context based on
            score threshold and returns all qualifying messages.
        Inherited Methods:
            Please refer to the FilterBase documentation for methods inherited by this class.

    """

    def __init__(self, score_threshold: Optional[float] = None, top_k: Optional[int] = None):
        """
        Initializes a new instance of the class with optional scoring threshold and top-k settings.
        The initialization sets up the instance with the provided score threshold and an optional top-k parameter which may be used to limit the
        number of items processed or retrieved. If no arguments are provided, both are set to their default values of None.
        
        Args:
            score_threshold (Optional[float]):
                 The minimum score that an item must achieve to be considered relevant. Defaults to None.
            top_k (Optional[int]):
                 The maximum number of top scoring items to consider. Defaults to None.
            

        """
        super().__init__()
        self._score_threshold = score_threshold
        self._top_k = top_k

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        
        Returns a list of scored chat messages after potentially applying a top-k filter.
            This method processes the given context by filtering through it and then optionally
            slicing the resulting list based on a top-k threshold if such a threshold has been set.
        
        Args:
            context (AgentContext):
                 The context of the agent's environment, potentially
                containing information relevant to filter and score chat messages.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of scored chat messages that may be limited to the top-k
                messages if '_top_k' attribute is set and greater than 0. If '_top_k' is None or
                not greater than 0, all filtered messages are returned.

        """
        filtered = self._filter(context)
        if self._top_k is not None and self._top_k > 0:
            return filtered[: self._top_k]

        return filtered

    def _filter(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Filters the chat messages from the evaluation context based on a score threshold.
        This method evaluates all the chat messages within the context's evaluation results. If a score threshold is set
        for the instance, it filters out the messages that have a score below the threshold. If no threshold is set,
        it returns all the messages.
        
        Args:
            context (AgentContext):
                 The context instance containing evaluation results for chat messages.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of chat messages that meet the score threshold criteria.
                This list may be empty if the evaluation results are None or if no messages meet the threshold.
            

        """
        all_eval_results = context.evaluation
        if all_eval_results is None:
            return []
        if self._score_threshold is not None:
            filtered = [x for x in all_eval_results if x.score >= self._score_threshold]
            return list(filtered)
        return list(all_eval_results)
