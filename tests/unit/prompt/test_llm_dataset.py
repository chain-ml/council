import json
import os
import unittest
from typing import List, Dict, Any

from tempfile import TemporaryDirectory

from council.prompt import LLMDatasetObject

from tests import get_data_filename
from .. import LLMDatasets


class TestLLMDataset(unittest.TestCase):
    def test_llm_dataset_from_yaml(self):
        filename = get_data_filename(LLMDatasets.sample)
        actual = LLMDatasetObject.from_yaml(filename)

        assert isinstance(actual, LLMDatasetObject)
        assert actual.kind == "LLMDataset"

    def test_jsonl_save(self):
        filename = get_data_filename(LLMDatasets.sample)
        dataset = LLMDatasetObject.from_yaml(filename)

        with TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "dataset.jsonl")
            dataset.save_jsonl(output_path)

            self.assertTrue(os.path.exists(output_path))
            saved_data = self._read_jsonl(output_path)
            self.assertEqual(len(saved_data), len(dataset.conversations))

            for entry in saved_data:
                self.assertIn("messages", entry)
                self.assertIsInstance(entry["messages"], list)

                for message in entry["messages"]:
                    self.assertIn("role", message)
                    self.assertIn("content", message)
                    self.assertIsInstance(message["role"], str)
                    self.assertIsInstance(message["content"], str)

            if dataset.system_prompt:
                for entry in saved_data:
                    self.assertEqual(entry["messages"][0]["role"], "system")
                    self.assertEqual(entry["messages"][0]["content"], dataset.system_prompt)

    def test_jsonl_save_with_split(self):
        filename = get_data_filename(LLMDatasets.sample)
        dataset = LLMDatasetObject.from_yaml(filename)

        with TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, "dataset.jsonl")
            val_split = 0.2
            dataset.save_jsonl(base_path, random_seed=1, val_split=val_split)

            train_path = os.path.join(tmp_dir, "dataset_train.jsonl")
            val_path = os.path.join(tmp_dir, "dataset_val.jsonl")

            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))

            train_data = self._read_jsonl(train_path)
            val_data = self._read_jsonl(val_path)

            total_conversations = len(dataset.conversations)
            expected_val_size = int(total_conversations * val_split)
            expected_train_size = total_conversations - expected_val_size

            self.assertEqual(len(train_data), expected_train_size)
            self.assertEqual(len(val_data), expected_val_size)

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        """Helper method to read JSONL file"""
        # TODO: move to LLMDataset
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
