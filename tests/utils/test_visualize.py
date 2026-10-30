# SPDX-License-Identifier: Apache-2.0

"""Tests for HTML and JavaScript escaping in model visualization."""

import html
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from onnx import TensorProto, helper

from onnxmltools.utils.visualize import visualize_model

# Payloads from the visualize_model injection reports: a graph input name that
# breaks out of the dagre-d3 setNode literal, and a producer name that injects
# markup into the heading.
_MALICIOUS_INPUT = "x'});alert(1);//</script><b>"
_MALICIOUS_PRODUCER = "<img src=x onerror=alert(2)>"
_MALICIOUS_VERSION = "<script>alert(3)</script>"
_MALICIOUS_OUTPUT = "a&b<c>"

_JS_LITERAL = r'"(?:\\.|[^"\\])*"'
_SCRIPT_OPEN = re.compile(r"<script\b", re.IGNORECASE)
_SCRIPT_CLOSE = re.compile(r"</script\s*>", re.IGNORECASE)


def _make_model(input_name, output_name, producer_name, producer_version):
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "visualize-test",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])],
    )
    return helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
    )


def _render(model):
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.getcwd()
        os.chdir(tmp)
        try:
            with patch("onnxmltools.utils.visualize.open_new_tab") as open_new_tab:
                visualize_model(model, dest="index.html")
                open_new_tab.assert_called_once()
            with open("index.html", encoding="utf-8") as handle:
                return handle.read()
        finally:
            os.chdir(previous)


def _js_fields(page, field):
    pattern = re.compile(rf"{field}:\s*({_JS_LITERAL})")
    return [json.loads(literal) for literal in pattern.findall(page)]


def _assert_one_close_per_script(testcase, page):
    opens = _SCRIPT_OPEN.findall(page)
    closes = _SCRIPT_CLOSE.findall(page)
    # d3, dagre-d3, and the inline graph script.
    testcase.assertEqual(len(opens), 3)
    testcase.assertEqual(len(closes), len(opens))


class TestVisualizeEscaping(unittest.TestCase):
    def test_malicious_names_and_producer_are_escaped(self):
        model = _make_model(
            _MALICIOUS_INPUT,
            _MALICIOUS_OUTPUT,
            _MALICIOUS_PRODUCER,
            _MALICIOUS_VERSION,
        )
        page = _render(model)

        self.assertNotIn(_MALICIOUS_INPUT, page)
        self.assertNotIn(_MALICIOUS_PRODUCER, page)
        self.assertNotIn(_MALICIOUS_VERSION, page)
        self.assertNotIn(_MALICIOUS_OUTPUT, page)
        self.assertNotIn("<img", page.lower())
        self.assertNotIn("<b>", page.lower())
        self.assertNotIn("<script>", page.lower())
        self.assertNotIn("</script><b>", page.lower())
        self.assertIn(r"\u003c/script\u003e", page)
        self.assertIn(r"\u003cb\u003e", page)
        self.assertIn(r"\u0026", page)

        heading = (
            "Model produced by: "
            + html.escape(_MALICIOUS_PRODUCER, quote=True)
            + " version("
            + html.escape(_MALICIOUS_VERSION, quote=True)
            + ")"
        )
        self.assertIn(f"<h2>{heading}</h2>", page)
        _assert_one_close_per_script(self, page)

        self.assertEqual(
            _js_fields(page, "label"),
            ["Identity", _MALICIOUS_INPUT, _MALICIOUS_OUTPUT],
        )
        classes = _js_fields(page, "class")
        self.assertEqual(classes[0], "type-Identity")
        expected_input_class = "type-" + re.sub(r"[^0-9A-Za-z_-]", "_", _MALICIOUS_INPUT)
        expected_output_class = "type-" + re.sub(r"[^0-9A-Za-z_-]", "_", _MALICIOUS_OUTPUT)
        self.assertEqual(classes[1], expected_input_class)
        self.assertEqual(classes[2], expected_output_class)
        self.assertNotIn("<", classes[1])
        self.assertNotIn("'", classes[1])

        inline = page.split('<script id="js">', 1)[1].split("</script>", 1)[0]
        without_strings = re.sub(_JS_LITERAL, '""', inline)
        self.assertNotIn("alert", without_strings)
        self.assertNotIn("<", without_strings)

    def test_plain_names_round_trip(self):
        model = _make_model("model_input", "model_output", "onnxmltools", "1.0")
        page = _render(model)

        self.assertIn("<h2>Model produced by: onnxmltools version(1.0)</h2>", page)
        self.assertNotIn("&lt;", page)
        self.assertNotIn(r"\u003c", page)
        self.assertIn(
            'g.setNode(0, { label: "Identity", class: "type-Identity" });',
            page,
        )
        self.assertEqual(
            _js_fields(page, "label"),
            ["Identity", "model_input", "model_output"],
        )
        self.assertEqual(
            _js_fields(page, "class"),
            ["type-Identity", "type-model_input", "type-model_output"],
        )
        _assert_one_close_per_script(self, page)


if __name__ == "__main__":
    unittest.main()
