# SPDX-License-Identifier: Apache-2.0

import html
import json
import os
import re
import shutil
import sys
from webbrowser import open_new_tab

# Characters that must not appear raw inside an inline <script> element.
_JS_HTML_ESCAPES = (
    ("<", r"\u003c"),
    (">", r"\u003e"),
    ("&", r"\u0026"),
)
_CSS_CLASS_UNSAFE = re.compile(r"[^0-9A-Za-z_-]")


def _js_string(value):
    """Return a JavaScript string literal safe to embed in an HTML script.

    ``json.dumps`` supplies the quotes and JavaScript escapes. ``<``, ``>``,
    and ``&`` are then written as Unicode escapes so the literal cannot close
    the surrounding script element or introduce an HTML character reference.
    """
    literal = json.dumps(str(value), ensure_ascii=True)
    for char, escaped in _JS_HTML_ESCAPES:
        literal = literal.replace(char, escaped)
    return literal


def _css_class(name):
    """Return a single CSS class token derived from a model string."""
    return "type-" + _CSS_CLASS_UNSAFE.sub("_", str(name))


def get_set_node(node, i="0"):
    return (
        "g.setNode("
        + str(i)
        + ", { label: "
        + _js_string(node)
        + ", class: "
        + _js_string(_css_class(node))
        + " });"
    )


def get_set_edge(start, end):
    return "g.setEdge(" + str(start) + ", " + str(end) + ");"


def get_nodes(graph):
    graph_nodes = [(i, node.op_type) for i, node in enumerate(graph.node, 0)]
    graph_nodes.extend(
        [(i, node.name) for i, node in enumerate(graph.input, len(graph_nodes))]
    )
    graph_nodes.extend(
        [(i, node.name) for i, node in enumerate(graph.output, len(graph_nodes) + 1)]
    )
    return graph_nodes


def get_nodes_builder(graph_nodes):
    _ret = [get_set_node(node[1], node[0]) for node in graph_nodes]
    return _ret


def get_edges(graph):
    nodes = graph.node
    initializer_names = [init.name for init in graph.initializer]
    output_node_hash = {}
    edge_list = []
    for i, node in enumerate(nodes, 0):
        for output in node.output:
            if output in output_node_hash.keys():
                output_node_hash[output].append(i)
            else:
                output_node_hash[output] = [i]
    for i, inp in enumerate(graph.input, len(nodes)):
        output_node_hash[inp.name] = [i]
    for i, node in enumerate(nodes, 0):
        for input in node.input:
            if input in output_node_hash.keys():
                edge_list.extend([(node_id, i) for node_id in output_node_hash[input]])
            else:
                if input not in initializer_names:
                    print("No corresponding output found for {0}.".format(input))
    for i, output in enumerate(graph.output, len(nodes) + len(graph.input) + 1):
        if output.name in output_node_hash.keys():
            edge_list.extend([(node_id, i) for node_id in output_node_hash[output.name]])
        else:
            pass
    return edge_list


def visualize_model(onnx_model, open_browser=True, dest="index.html"):
    """
    Creates a graph visualization of an ONNX protobuf model.
    It creates a SVG graph with *d3.js* and stores it into a file.

    :param model: ONNX model (protobuf object)
    :param open_browser: opens the browser
    :param dest: destination file

    Example:

    ::

        from onnxmltools.utils import visualize_model
        visualize_model(model)
    """
    graph = onnx_model.graph
    model_info = (
        "Model produced by: "
        + html.escape(onnx_model.producer_name, quote=True)
        + " version("
        + html.escape(onnx_model.producer_version, quote=True)
        + ")"
    )

    html_str = """
    <!doctype html>
    <meta charset="utf-8">
    <title>ONNX Visualization</title>
    <script src="https://d3js.org/d3.v3.min.js"></script>
    <link rel="stylesheet" href="styles.css">
    <script src="dagre-d3.min.js"></script>

    <h2>[model_info]</h2>

    <svg id="svg-canvas" width=960 height=600></svg>

    <script id="js">
    var g = new dagreD3.graphlib.Graph()
    .setGraph({})
    .setDefaultEdgeLabel(function() { return {}; });

    [nodes_html]

    g.nodes().forEach(function(v) {
    var node = g.node(v);
    // Round the corners of the nodes
    node.rx = node.ry = 5;
    });

    [edges_html]

    // Create the renderer
    var render = new dagreD3.render();

    // Set up an SVG group so that we can translate the final graph.
    var svg = d3.select("svg"),
        svgGroup = svg.append("g");

    // Run the renderer. This is what draws the final graph.
    render(d3.select("svg g"), g);

    // Center the graph
    svgGroup.attr("transform", "translate(20, 20)");
    svg.attr("height", g.graph().height + 40);
    svg.attr("width", g.graph().width + 40);
    </script>
    """

    html_str = html_str.replace("[nodes_html]", "\n".join(get_nodes_builder(get_nodes(graph))))

    html_str = html_str.replace(
        "[edges_html]",
        "\n".join([get_set_edge(edge[0], edge[1]) for edge in get_edges(graph)]),
    )

    html_str = html_str.replace("[model_info]", model_info)

    with open(dest, "w", encoding="utf-8") as html_file:
        html_file.write(html_str)

    pkgdir = sys.modules["onnxmltools"].__path__[0]
    fullpath = os.path.join(pkgdir, "utils", "styles.css")
    shutil.copy(fullpath, os.getcwd())
    fullpath = os.path.join(pkgdir, "utils", "dagre-d3.min.js")
    shutil.copy(fullpath, os.getcwd())

    open_new_tab("file://" + os.path.realpath("index.html"))
