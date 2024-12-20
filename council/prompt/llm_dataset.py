from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, List, Mapping, Optional, Sequence

import yaml
from council.llm.base import LLMMessage
from council.llm.llm_function import LLMResponse
from council.utils import DataObject, DataObjectSpecBase


class LLMDatasetConversation:
    """
    Represents a conversation between user and assistant with optional labels.
    """

    def __init__(self, messages: Sequence[LLMMessage], labels: Optional[Mapping[str, str]]):
        self.messages = list(messages)
        self.labels: Dict[str, str] = dict(labels) if labels is not None else {}

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMDatasetConversation:
        messages = values.get("messages", [])
        if not messages:
            raise ValueError("Conversation must contain at least one message")
        llm_messages = [LLMMessage.from_dict(message) for message in messages]
        labels = values.get("labels")
        return LLMDatasetConversation(llm_messages, labels)

    @classmethod
    def from_llm_response(
        cls, response: LLMResponse, labels: Optional[Mapping[str, str]] = None
    ) -> LLMDatasetConversation:
        return LLMDatasetConversation(response.to_messages(), labels)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"messages": [message.to_dict() for message in self.messages]}
        if self.labels:
            result["labels"] = self.labels
        return result

    def format(self, start_prefix: str, end_prefix: str) -> str:
        """Format conversation as a few shot example."""

        parts = [start_prefix]
        parts.extend([f"{message}" for message in self.messages])
        parts.append(end_prefix)

        return "\n".join(parts)

    @staticmethod
    def get_message_pair(*, user: str, assistant: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class LLMDatasetSpec(DataObjectSpecBase):
    def __init__(self, conversations: List[LLMDatasetConversation], system_prompt: Optional[str] = None) -> None:
        self.conversations = conversations
        self.system_prompt = system_prompt.strip() if system_prompt is not None else None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LLMDatasetSpec:
        conversations = values.get("conversations", [])
        if not conversations:
            raise ValueError("Dataset must contain at least one conversation")

        parsed_conversations = [LLMDatasetConversation.from_dict(c) for c in conversations]
        system_prompt = values.get("system_prompt")
        return LLMDatasetSpec(parsed_conversations, system_prompt)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"conversations": [conv.to_dict() for conv in self.conversations]}
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        return result

    def __str__(self):
        result = f"{len(self.conversations)} conversation(s)"
        if self.system_prompt is not None:
            result += " with shared system prompt"
        return result


class LLMDatasetObject(DataObject[LLMDatasetSpec]):
    """
    Helper class to instantiate a LLMDataset from a YAML file.

    LLMDataset represents a dataset to be used for fine-tuning / batch API or managing few shot examples.
    Contains a list of conversations between user and assistant and optional shared system prompt;
    if specified, it will be a system prompt for every conversation in the dataset.
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMDatasetObject:
        return super()._from_dict(LLMDatasetSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMDatasetObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMDataset")
            return LLMDatasetObject.from_dict(values)

    @property
    def system_prompt(self) -> Optional[str]:
        """Return system prompt if any."""
        return self.spec.system_prompt

    @property
    def conversations(self) -> List[LLMDatasetConversation]:
        """Return all raw conversations in the dataset."""
        return self.spec.conversations

    def count_labels(self) -> DefaultDict[str, Counter]:
        """
        Count occurrences of each label value grouped by label key.
        Returns a dictionary where keys are label names and values are Counters of label values.
        """
        label_counters: DefaultDict[str, Counter] = defaultdict(Counter)
        for conversation in self.conversations:
            if conversation.labels:
                for label_key, label_value in conversation.labels.items():
                    label_counters[label_key][label_value] += 1
        return label_counters

    def to_jsonl_messages(self) -> List[Dict[str, List[Dict[str, str]]]]:
        """
        Convert the dataset to JSONL format with OpenAI messages structure.
        Returns a list of dictionaries containing messages.
        """
        messages_starter = []
        if self.system_prompt is not None:
            messages_starter = [{"role": "system", "content": self.system_prompt}]

        jsonl_lines = []
        for conversation in self.conversations:
            messages = messages_starter + [msg.to_dict() for msg in conversation.messages]
            jsonl_lines.append({"messages": messages})

        return jsonl_lines

    def save_jsonl_messages(
        self, path: str, random_seed: Optional[int] = None, val_split: Optional[float] = None
    ) -> None:
        """
        Save the dataset as JSONL messages file(s), optionally splitting into training and validation sets.
        JSONL file then can be used for fine-tuning.
        See https://platform.openai.com/docs/guides/fine-tuning.

        Args:
            path: Base path for saving the file(s)
            random_seed: If provided, will be used to shuffle dataset before saving (default: None)
            val_split: If provided, fraction of data to use for validation and create separate files for train and val.
                       If None, saves all data to a single file (default: None)

        Examples:
            # Save all data into a single `my_dataset.jsonl` file
            dataset.save_jsonl("my_dataset.jsonl")  # Creates my_dataset.jsonl

            # Split into train/val sets (80/20 split) and saves into `my_dataset_train.jsonl` and `my_dataset_val.jsonl`
            dataset.save_jsonl("my_dataset.jsonl", random_seed=42, val_split=0.2)
        """

        jsonl_lines = self.to_jsonl_messages()
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(jsonl_lines)

        base_path = path[:-6] if path.endswith(".jsonl") else path

        if val_split is None:
            self._save_jsonl(f"{base_path}.jsonl", jsonl_lines)
            return

        split_index = int(len(jsonl_lines) * (1 - val_split))
        train_lines, val_lines = jsonl_lines[:split_index], jsonl_lines[split_index:]

        self._save_jsonl(f"{base_path}_train.jsonl", train_lines)
        self._save_jsonl(f"{base_path}_val.jsonl", val_lines)

    def save_jsonl_requests(self, path: str, model: str, url: str = "/v1/chat/completions") -> None:
        """
        Save the dataset as JSONL request file, which can be used for batch API.
        See https://platform.openai.com/docs/guides/batch.

        Args:
            path: Path to the output file
            model: OpenAI model name
            url: OpenAI API URL (default: "/v1/chat/completions")

        Examples:
            dataset.save_jsonl_request("my_batch.jsonl", "gpt-4o-mini")
        """
        messages_lines = self.to_jsonl_messages()

        request_lines = [
            {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": url,
                "body": {"model": model, "messages": message_line["messages"]},
            }
            for i, message_line in enumerate(messages_lines)
        ]

        self._save_jsonl(path, request_lines)

    def format_examples(self, start_prefix: str = "# Example {i}", end_prefix: str = "") -> List[str]:
        """
        Format dataset conversations as a few shot examples. Does not include system prompt.
        If `start_prefix` or `end_prefix` contain `{i}`, it will be replaced with the example number.
        """

        examples = []
        for i, conversation in enumerate(self.conversations, start=1):
            start_prefix_formatted = start_prefix.format(i=i) if "{i}" in start_prefix else start_prefix
            end_prefix_formatted = end_prefix.format(i=i) if "{i}" in end_prefix else end_prefix
            examples.append(conversation.format(start_prefix_formatted, end_prefix_formatted))

        return examples

    @staticmethod
    def _save_jsonl(filename: str, lines: List[Dict[str, Any]]) -> None:
        """Helper method to save lines to JSONL file."""
        with open(filename, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

    @staticmethod
    def read_jsonl(path: str) -> List[Dict[str, Any]]:
        """Helper method to read JSONL file into list of dictionaries."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data


class LLMDatasetValidationException(Exception):
    """Exception raised for validation errors in LLMDatasetObject."""


class LLMDatasetValidator:
    """
    Helper class to validate the content of LLMDatasetObject.
    """

    @staticmethod
    def validate_for_batch_api(dataset: LLMDatasetObject) -> None:
        """
        Validate dataset for batch API.

        Raises:
            LLMDatasetValidationException
                If dataset contains conversations that do not end with a user message.
        """

        for idx, conversation in enumerate(dataset.conversations, start=1):
            if conversation.messages[-1].role != "user":
                raise LLMDatasetValidationException(f"Conversation #{idx}: must end with a user message")

        print("All conversations end with a user message.")

    @staticmethod
    def validate_for_fine_tuning(dataset: LLMDatasetObject) -> None:
        """
        Validate dataset for fine-tuning.

        Raises:
            LLMDatasetValidationException
                If dataset contains conversations that does not follow the pattern:
                user -> assistant -> user -> assistant -> ...
        """

        for idx, conversation in enumerate(dataset.conversations, start=1):
            prefix = f"Conversation #{idx}:"

            if len(conversation.messages) % 2 != 0:
                raise LLMDatasetValidationException(f"{prefix} There must be an even number of messages")

            for i in range(0, len(conversation.messages), 2):
                if conversation.messages[i].role != "user":
                    raise LLMDatasetValidationException(f"{prefix} Message #{i} must be a user message")
                if conversation.messages[i + 1].role != "assistant":
                    raise LLMDatasetValidationException(f"{prefix} Message #{i + 1} must be an assistant message")

        print("All conversations have an even number of messages with alternating user/assistant roles.")
