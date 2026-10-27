"""Authentication regression without loading optional Llama/JAX dependencies."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SOURCE = Path(__file__).resolve().parents[1] / "moment/models/llmtime.py"


class LlamaAuthenticationTests(unittest.TestCase):
    def check_auth(self, environment, expected):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"llama2_model_string", "get_tokenizer", "get_model_and_tokenizer"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
        tokenizer.unk_token = "<unk>"
        tokenizer_class = MagicMock()
        tokenizer_class.from_pretrained.return_value = tokenizer
        model_class = MagicMock()
        namespace = {"os": os, "LlamaTokenizer": tokenizer_class,
                     "LlamaForCausalLM": model_class,
                     "torch": SimpleNamespace(float16="float16")}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        with patch.dict(os.environ, environment, clear=True):
            namespace["get_model_and_tokenizer"]("7b")
        self.assertTrue(tokenizer_class.from_pretrained.call_count == 1)
        self.assertTrue(model_class.from_pretrained.call_count == 1)
        # Boolean assertions deliberately never display a credential on failure.
        for loader in (tokenizer_class.from_pretrained, model_class.from_pretrained):
            supplied = loader.call_args.kwargs.get("token")
            self.assertTrue(supplied == expected, "Loader must use HF_TOKEN or cached-login fallback")

    def test_both_loaders_use_environment_token(self):
        self.check_auth({"HF_TOKEN": "fixture-value-from-environment"}, "fixture-value-from-environment")

    def test_unset_environment_allows_cached_login(self):
        self.check_auth({}, None)

    def test_empty_environment_allows_cached_login(self):
        self.check_auth({"HF_TOKEN": ""}, None)


if __name__ == "__main__":
    unittest.main()
