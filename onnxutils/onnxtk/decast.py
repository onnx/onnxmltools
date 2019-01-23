import sys
import onnx
from .optimizer import LinkedNode, Solution


def remove_cast(lnodes, op_set):

    sln = None
    while True:
        for n_ in lnodes:
            if n_ in op_set and n_.in_single_path():
                if n_.precedence[0].op_type == 'Cast' and n_.single_ouput().op_type == 'Cast':
                    sln = Solution(None, n_.precedence[0], n_.precedence[0], n_)
                    break

        if sln is None:
            break

        lnodes = sln.apply(lnodes)

    return lnodes


def decast(origin_model, oplist):
    """
    remove the ONNX cast op from the specified operator.
    :param origin_model:
    :param oplist:
    :return:
    """
    graph = origin_model.graph
    nodelist = list(graph.node)
    del graph.node[:]

    all_nodes = LinkedNode.build_from_onnx(nodelist, origin_model.opset_import[0].version,
                              inputs=graph.input,
                              outputs=graph.output)

    nodes = remove_cast(all_nodes, set(oplist))
    graph.node.extend([n_.generate for n_ in nodes])

    return origin_model


def main():
    if len(sys.argv) < 4:
        print('decast.py model_in  model_out <op1, ...>')
        return

    input = sys.argv[1]
    output = sys.argv[2]
    op_list = sys.argv[2:]


    oxml = onnx.load_model(input)
    oxml = decast(oxml, op_list)
    onnx.save_model(oxml, output)


if __name__ == "__main__":
    main()
