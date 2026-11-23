# SPDX-License-Identifier: Apache-2.0

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from onnx import TensorProto, helper
from onnxmltools.utils import visualize_model


class VisualizationHTML(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.tags = []
        self.scripts = []
        self.heading = []
        self.in_script = False
        self.in_heading = False
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "script":
            self.in_script = True
            self.scripts.append("")
        elif tag == "h2":
            self.in_heading = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        elif tag == "h2":
            self.in_heading = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts[-1] += data
        elif self.in_heading:
            self.heading.append(data)


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def model(input_name="X", output_name="Y", op_type="Identity"):
        graph = helper.make_graph(
            [helper.make_node(op_type, [input_name], [output_name])],
            "graph",
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, 1])],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, 1])],
        )
        return helper.make_model(graph, producer_name="producer", producer_version="1")

    def render(self, model, open_browser=False, dest=None):
        destination = dest or self.root / "index.html"
        with patch("onnxmltools.utils.visualize.open_new_tab") as browser:
            visualize_model(model, open_browser=open_browser, dest=destination)
        return destination.read_text(encoding="utf-8"), browser

    def test_normal_model_renders_labels_and_edges(self):
        content, _ = self.render(self.model())
        self.assertIn("Identity", content)
        self.assertIn("g.setEdge(1, 0);", content)
        self.assertIn("g.setEdge(0, 3);", content)
        self.assertEqual(
            "".join(VisualizationHTML(content).heading),
            "Model produced by: producer version(1)",
        )

    def test_producer_metadata_is_text(self):
        payloads = [
            "<script>globalThis.injected = true;</script>",
            '<img src=x onerror="globalThis.injected = true">',
            "'\" & < > \\ \n \u2028 \u2029 caf\u00e9 $nodes_html [nodes_html]",
        ]
        for field in ("producer_name", "producer_version"):
            for payload in payloads:
                with self.subTest(field=field, payload=payload):
                    model = self.model()
                    setattr(model, field, payload)
                    content, _ = self.render(model)
                    document = VisualizationHTML(content)
                    self.assertEqual(document.tags.count("script"), 3)
                    self.assertNotIn("img", document.tags)
                    self.assertEqual(
                        "".join(document.heading),
                        f"Model produced by: {model.producer_name} "
                        f"version({model.producer_version})",
                    )

    def test_graph_labels_round_trip_as_script_safe_data(self):
        payloads = [
            "'); globalThis.injected = true; //",
            "</ScRiPt><script>globalThis.injected = true;</script>",
            '<svg onload=alert(1)> & \\ " \n \u2028 \u2029 caf\u00e9',
            "$model_info [model_info] $edges_html [edges_html]",
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                model = self.model(payload, payload + "_out", payload + "_op")
                model.producer_name = 'producer\'s "name"\n[edges_html]'
                content, _ = self.render(model)
                document = VisualizationHTML(content)
                self.assertEqual(document.tags.count("script"), 3)
                self.assertEqual(document.tags.count("svg"), 1)
                script = document.scripts[-1]
                self.assertNotIn("</script", script.lower())
                nodes = re.findall(r"g\.setNode\((\d+), (.+)\);", script)
                self.assertEqual(len(nodes), 3)
                labels = []
                for _, attributes in nodes:
                    data = json.loads(attributes)
                    labels.append(data["label"])
                    self.assertEqual(data["class"], "type-" + data["label"])
                self.assertEqual(labels, [payload + "_op", payload, payload + "_out"])

    def test_browser_is_not_opened_when_disabled(self):
        _, browser = self.render(self.model(), open_browser=False)
        browser.assert_not_called()

    def test_browser_and_assets_use_the_selected_destination(self):
        folder = self.root / "nested space"
        folder.mkdir()
        destination = folder / "model #1.html"
        _, browser = self.render(self.model(), open_browser=True, dest=destination)
        browser.assert_called_once_with(destination.resolve().as_uri())
        for asset in ("styles.css", "dagre-d3.min.js"):
            self.assertTrue((folder / asset).is_file())


if __name__ == "__main__":
    unittest.main()
