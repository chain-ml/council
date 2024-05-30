import unittest

from council.utils.data_object import DataObjectMetadata


class TestCodeDataObject(unittest.TestCase):

    def test_code_data_object(self):
        labels = {"test": "value1", "test2": "value2", "test_array": ["value31", "value32", "value33"]}
        metadata = DataObjectMetadata("test", labels=labels)
        self.assertEqual(metadata.labels, labels.copy())
        self.assertTrue(metadata.has_label("test"))

        self.assertTrue(metadata.is_matching_labels(labels={"test": "value1"}))
        self.assertTrue(metadata.is_matching_labels(labels={"test": "value1", "test2": "value2", "test_array": None}))
        self.assertTrue(metadata.is_matching_labels(labels={"test": None}))
        self.assertTrue(metadata.is_matching_labels(labels={"test_array": None}))
        self.assertTrue(metadata.is_matching_labels(labels={"test_array": ["value31", "value33"]}))

        self.assertFalse(metadata.is_matching_labels(labels={"test": "value2"}))
        self.assertFalse(metadata.is_matching_labels(labels={"key_not_exist": None}))
        self.assertFalse(metadata.is_matching_labels(labels={"key_not_exist": "value2"}))
        self.assertFalse(metadata.is_matching_labels(labels={"test_array": "value32"}))
        self.assertFalse(metadata.is_matching_labels(labels={"test_array": ["value31", "value33", "value34"]}))
