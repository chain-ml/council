import unittest
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class TestNotebooks(unittest.TestCase):
    def _test_notebook(self, filepath: str, timeout: int = 30):
        module_path = os.path.dirname(__file__)
        file_path = os.path.join(module_path, "..", "..", "docs", "source", filepath)
        with open(file_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=timeout)
            ep.preprocess(nb)

    def test_first_example(self):
        self._test_notebook("getting_started/first_example.ipynb", timeout=10)

    @unittest.skip("indexing takes too long")
    def test_llamaindex(self):
        self._test_notebook("examples/integrations/llamaindex_integration.ipynb")

    def test_langchain_llm(self):
        self._test_notebook("examples/integrations/langchain_llm_integration.ipynb")

    def test_multi_chain_agent(self):
        self._test_notebook("examples/multi_chain_agent.ipynb", timeout=60)
