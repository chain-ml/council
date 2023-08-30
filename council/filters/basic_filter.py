from typing import List, Optional

from council.contexts import AgentContext, ScoredChatMessage
from council.filters import FilterBase
from council.runners import Budget


class BasicFilter(FilterBase):
    def __init__(self, score_threshold: Optional[float] = None, top_k: Optional[int] = None):
        self._score_threshold = score_threshold
        self._top_k = top_k

    def _execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        filtered = self._filter(context)
        if self._top_k is not None and self._top_k > 0:
            return filtered[: self._top_k]

        return filtered

    def _filter(self, context: AgentContext):
        all_eval_results = context.last_evaluator_iteration()
        if all_eval_results is None:
            return []
        if self._score_threshold is not None:
            filtered = [x for x in all_eval_results if x.score >= self._score_threshold]
            return list(filtered)
        return all_eval_results
