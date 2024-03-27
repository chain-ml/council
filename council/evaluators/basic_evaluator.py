"""

A module that provides the BasicEvaluator, which is a concrete implementation of the EvaluatorBase class.

This module defines the BasicEvaluator class which inherits from EvaluatorBase. The BasicEvaluator
implements the functionality to evaluate messages within a given context and assigns them a score
based on predefined criteria. The evaluator is intended to be part of a larger system where messages
in a conversation chain are analyzed and scored to determine an appropriate response or action.

Classes:
    BasicEvaluator: Concrete implementation of the abstract EvaluatorBase class. Provides the
    mechanism to examine messages from a conversation context and scores them.

Typical usage example:

    evaluator = BasicEvaluator()
    scored_messages = evaluator.execute(context)


"""
from typing import List

from council.contexts import (
    AgentContext,
    ScoredChatMessage,
    ChatMessage,
)
from .evaluator_base import EvaluatorBase


class BasicEvaluator(EvaluatorBase):
    """
    A subclass of `EvaluatorBase` that evaluates messages in a given context by assigning a basic score based on certain criteria.
    This evaluator goes through messages in the context's chains and calculates a score for the last message in each chain. It gives a score of 1.0 if the last message is from a skill and is not an error, else a score of 0.0 is given. These scores are encapsulated along with the messages into `ScoredChatMessage` objects and returned as a list.
    
    Attributes:
        Inherited from EvaluatorBase, no additional attributes.
    
    Methods:
        _execute(context:
             AgentContext) -> List[ScoredChatMessage]:
            The method that performs the evaluation of messages within the provided context. It overrides the `_execute` method from the EvaluatorBase class.
    
    Args:
        context (AgentContext):
             The context containing chains of messages that need to be evaluated.
    
    Returns:
        (List[ScoredChatMessage]):
             A list of `ScoredChatMessage` objects that contains evaluated messages along with their calculated scores.

    """

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Performs the execution of the agent within a given context, scoring chat messages based on certain conditions.
        This method iterates over the chat message chains in the provided context, evaluating and scoring the last message of each chain based on whether it is from a kind skill and was processed without errors. Messages meeting these criteria receive a score of 1.0, while all others are scored as 0.0. Each scored message is encapsulated in a `ScoredChatMessage` object and added to a result list that is returned after all chains have been processed.
        
        Args:
            context (AgentContext):
                 The context in which the agent is operating, including chat message chains and other relevant data.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of scored chat messages, where each `ScoredChatMessage` contains a `ChatMessage` and its associated score.

        """
        result: List[ScoredChatMessage] = []
        for chain_messages in context.chains:
            chain_result = chain_messages.last_message
            if chain_result is None:
                continue
            score = 1.0 if chain_result.is_kind_skill and chain_result.is_ok else 0.0
            result.append(
                ScoredChatMessage(
                    ChatMessage.agent(
                        message=chain_result.message,
                        data=chain_result.data,
                        source=chain_result.source,
                        is_error=chain_result.is_error,
                    ),
                    score,
                )
            )
        return result
