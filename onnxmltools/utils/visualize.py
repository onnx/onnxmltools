# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from webbrowser import open_new_tab


def get_set_node(node, i="0"):
    return "g.setNode(" + str(i) + ", { label: '" + node + "', class: 'type-" + node + "' });"


def get_set_edge(start, end):
    return "g.setEdge(" + str(start) + ", " + str(end) + ");"


def get_nodes(graph):
    graph_nodes = [(i, node.op_type) for i, node in enumerate(graph.node, 0)]
    graph_nodes.extend([(i, node.name)
                        for i, node in enumerate(graph.input, len(graph_nodes))])
    graph_nodes.extend([(i, node.name)
                        for i, node in enumerate(graph.output, len(graph_nodes) + 1)])
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
                edge_list.extend([(node_id, i)
                                  for node_id in output_node_hash[input]])
            else:
                if not input in initializer_names:
                    print(
                        "No corresponding output found for {0}.".format(input))
    for i, output in enumerate(graph.output, len(nodes) + len(graph.input) + 1):
        if output.name in output_node_hash.keys():
            edge_list.extend([(node_id, i)
                              for node_id in output_node_hash[output.name]])
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
    model_info = "Model produced by: " + onnx_model.producer_name + \
        " version(" + onnx_model.producer_version + ")"

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

    html_str = html_str.replace("[nodes_html]", "\n".join(
        get_nodes_builder(get_nodes(graph))))

    html_str = html_str.replace("[edges_html]", "\n".join(
        [get_set_edge(edge[0], edge[1]) for edge in get_edges(graph)]))

    html_str = html_str.replace("[model_info]", model_info)

    Html_file = open(dest, "w")
    Html_file.write(html_str)
    Html_file.close()

    pkgdir = sys.modules['onnxmltools'].__path__[0]
    fullpath = os.path.join(pkgdir, "utils", "styles.css")
    shutil.copy(fullpath, os.getcwd())
    fullpath = os.path.join(pkgdir, "utils", "dagre-d3.min.js")
    shutil.copy(fullpath, os.getcwd())

    open_new_tab("file://" + os.path.realpath("index.html"))
