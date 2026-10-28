import importlib.util
import types
import sys
import unittest
import json
import io
from pathlib import Path
from unittest.mock import patch


class ImagesTests(unittest.TestCase):
    def test_transport_and_text_compatibility(self):
        stub = types.ModuleType("habitat_llm.llm.base_llm")
        stub.BaseLLM = object
        spec = importlib.util.spec_from_file_location(
            "adapter",
            Path(__file__).resolve().parents[1] / "habitat_llm/llm/local_chat.py",
        )
        m = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"habitat_llm.llm.base_llm": stub}):
            spec.loader.exec_module(m)
        self.assertEqual(m.image_content("hello"), "hello")
        for invalid in [[], "file.png", [b"not an image"]]:
            with self.assertRaises(ValueError):
                m.image_content("hello", invalid)
        obj = object.__new__(m.LocalChat)
        obj.generation_params = types.SimpleNamespace(
            model="qwen", temperature=0, top_p=1, max_tokens=32
        )
        obj.llm_conf = types.SimpleNamespace(system_message="describe")
        obj.url, obj.timeout = "http://localhost/v1/chat/completions", 10
        captured = []

        def receive(request, timeout):
            captured.append(json.loads(request.data))
            return io.BytesIO(
                b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'
            )

        obj.http = types.SimpleNamespace(open=receive)
        self.assertEqual(
            obj.generate("describe", images=[b"\x89PNG\r\n\x1a\nexample"]), "ok"
        )
        parts = captured[0]["messages"][1]["content"]
        self.assertEqual(parts[0]["text"], "describe")
        self.assertTrue(
            parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        obj.generate("text only")
        self.assertEqual(captured[1]["messages"][1]["content"], "text only")
