import os
import unittest

from tempfile import TemporaryDirectory

from council.prompt import LLMDatasetObject

from tests import get_data_filename
from .. import LLMDatasets


class TestLLMDataset(unittest.TestCase):
    def _validate_messages(self, messages):
        for message in messages:
            self.assertIn("role", message)
            self.assertIn("content", message)
            self.assertIsInstance(message["role"], str)
            self.assertIsInstance(message["content"], str)

    def test_llm_dataset_from_yaml(self):
        filename = get_data_filename(LLMDatasets.sample)
        actual = LLMDatasetObject.from_yaml(filename)

        assert isinstance(actual, LLMDatasetObject)
        assert actual.kind == "LLMDataset"

    def test_save_jsonl_messages(self):
        filename = get_data_filename(LLMDatasets.sample)
        dataset = LLMDatasetObject.from_yaml(filename)

        with TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "dataset.jsonl")
            dataset.save_jsonl_messages(output_path)

            self.assertTrue(os.path.exists(output_path))
            saved_data = LLMDatasetObject.read_jsonl(output_path)
            self.assertEqual(len(saved_data), len(dataset.conversations))

            for entry in saved_data:
                self.assertIn("messages", entry)
                self.assertIsInstance(entry["messages"], list)
                self._validate_messages(entry["messages"])

            if dataset.system_prompt:
                for entry in saved_data:
                    self.assertEqual(entry["messages"][0]["role"], "system")
                    self.assertEqual(entry["messages"][0]["content"], dataset.system_prompt)

    def test_save_jsonl_messages_with_split(self):
        filename = get_data_filename(LLMDatasets.sample)
        dataset = LLMDatasetObject.from_yaml(filename)

        with TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, "dataset.jsonl")
            val_split = 0.2
            dataset.save_jsonl_messages(base_path, random_seed=1, val_split=val_split)

            train_path = os.path.join(tmp_dir, "dataset_train.jsonl")
            val_path = os.path.join(tmp_dir, "dataset_val.jsonl")

            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(val_path))

            train_data = LLMDatasetObject.read_jsonl(train_path)
            val_data = LLMDatasetObject.read_jsonl(val_path)

            total_conversations = len(dataset.conversations)
            expected_val_size = int(total_conversations * val_split)
            expected_train_size = total_conversations - expected_val_size

            self.assertEqual(len(train_data), expected_train_size)
            self.assertEqual(len(val_data), expected_val_size)

    def test_save_jsonl_request(self):
        filename = get_data_filename(LLMDatasets.sample)
        dataset = LLMDatasetObject.from_yaml(filename)

        with TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "batch.jsonl")
            dataset.save_jsonl_requests(output_path, "gpt-4o-mini")

            self.assertTrue(os.path.exists(output_path))
            saved_data = LLMDatasetObject.read_jsonl(output_path)
            self.assertEqual(len(saved_data), len(dataset.conversations))

            for entry in saved_data:
                self.assertIn("custom_id", entry)
                self.assertIn("method", entry)
                self.assertIn("url", entry)
                self.assertIn("body", entry)

                self.assertEqual(entry["method"], "POST")
                self.assertEqual(entry["url"], "/v1/chat/completions")

                body = entry["body"]
                self.assertIn("model", body)
                self.assertEqual(body["model"], "gpt-4o-mini")
                self.assertIn("messages", body)
                self.assertIsInstance(body["messages"], list)
                self._validate_messages(body["messages"])
