# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import six
from onnxmltools.proto import onnx_proto, helper


class LinkedNode(object):

    def __init__(self, node=None):
        self.origin = node  # type: onnx_proto.NodeProto
        self.input = {} if node is None else {i_: i_ for i_ in node.input}
        self.output = {} if node is None else {o_: o_ for o_ in node.output}
        self.precedence = []
        self.successor = []
        self.attributes = {}

    @property
    def op_type(self):
        return None if self.origin is None else self.origin.op_type

    @property
    def is_identity(self):
        return False if self.origin is None else self.origin.op_type == 'Identity'

    @property
    def is_transpose(self):
        return False if self.origin is None else self.origin.op_type == 'Transpose'

    @property
    def in_single_path(self):
        return False if self.origin is None else len(self.origin.output) == 1

    @property
    def in_or_out(self):
        return self.origin is None

    @property
    def single_input(self):
        assert self.origin is not None and len(self.input) == 1
        return next(value for (key, value) in six.iteritems(self.input))

    @property
    def single_output(self):
        assert self.origin is not None and len(self.output) == 1
        return next(value for (key, value) in six.iteritems(self.output))

    def generate(self):
        if not self.input and not self.output and not self.attributes:
            return self.origin
        else:
            onode = helper.make_node(
                self.origin.op_type,
                [self.input.get(i_, i_) for i_ in self.origin.input],
                [self.output.get(o_, o_) for o_ in self.origin.output],
                self.origin.name,
                self.origin.doc_string,
            )

            onode.attribute.extend(
                helper.make_attribute(attr.name, self.attributes[attr.name])
                if attr.name in self.attributes else
                attr for attr in self.origin.attribute
            )

            return onode

    def add_precedence(self, pre, tname):
        self.precedence.append(pre)
        pre.successor.append(self)

    @staticmethod
    def build_from_onnx(onnx_nodes):
        view = []
        var_map = {}
        for o_ in onnx_nodes:
            ln = LinkedNode(o_)
            view.append(ln)
            for var_ in o_.output:
                assert var_map.get(var_) is None
                var_map.setdefault(var_, ln)

        for n_ in view:
            for var_ in n_.origin.input:
                target = var_map.get(var_)
                if target is None:
                    target = LinkedNode()  # create an empty node as input
                n_.add_precedence(target, var_)

        for n_ in view:  # add a dummy output node.
            if len(n_.successor) < 1:
                for var_ in n_.origin.output:
                    LinkedNode().add_precedence(n_, var_)

        return view


class Solution(object):
    def __init__(self, begin, begin_n, end_p, end):
        self.begin = begin
        self.begin_n = begin_n
        self.end_p = end_p
        self.end = end

    @staticmethod
    def get_perm(attrs):
        try:
            return next(
                helper.get_attribute_value(attr) for attr in attrs if attr.name == 'perm')
        except StopIteration:
            return None

    @staticmethod
    def delete_node(node_list, begin, node, end):
        if end.in_or_out:
            # if the end is output node, the output name will be kept to avoid the model output updating.
            begin.output[begin.origin.output[0]] = node.single_output
        else:
            target_var_name = node.single_input
            assert target_var_name in begin.origin.output # since the output info never be updated, except the final.
            end.input[node.origin.output[0]] = target_var_name

        begin.successor = [end if v_ == node else v_ for v_ in begin.successor]
        end.precedence = [begin if v_ == node else v_ for v_ in end.precedence]

        node_list.remove(node)
        return node_list

    def apply(self, node_list):
        node = self.begin_n  # type: LinkedNode
        while node != self.end:
            assert len(node.successor) == 1
            end = node.successor[0]
            node_list = self.delete_node(node_list, self.begin, node, end)
            node = end

        return node_list


class MergeSolution(Solution):
    def apply(self, node_list):
        perm0 = self.get_perm(self.begin_n.origin.attribute)
        perm1 = self.get_perm(self.end_p.origin.attribute)
        assert len(perm0) == len(perm1)
        perm_f = [perm0[idx] for idx in perm1]
        if perm_f == list(six.moves.range(len(perm_f))):
            super(MergeSolution, self).apply(node_list)  # delete all transpose nodes
        else:
            node_list = self.delete_node(node_list, self.begin_n, self.end_p, self.end)
            self.begin_n.attribute['perm'] = perm_f
        return node_list


class RedundantOptimizer(object):
    def __init__(self):
        pass

    @staticmethod
    def is_useless_transpose(attrs):
        perm = Solution.get_perm(attrs)
        return perm == range(len(perm))

    def find(self, node_list):
        for n_ in node_list:
            if n_.is_identity and n_.in_single_path:
                end = n_.successor[0]
                end_pre = n_
                while end is not None and end.is_identity and n_.in_single_path:
                    end_pre = end
                    end = end.successor[0]
                solution = Solution(n_.precedence[0], n_, end_pre, end)
                return solution
            elif n_.is_transpose and \
                n_.in_single_path and \
                n_.successor[0].is_transpose and \
                    n_.successor[0].in_single_path:
                solution = MergeSolution(n_.precedence[0], n_, n_.successor[0], n_.successor[0].successor[0])
                return solution
            elif n_.is_transpose and \
                n_.in_single_path and \
                    self.is_useless_transpose(n_.attribute):
                solution = Solution(n_.precedence[0], n_, n_, n_.successor[0])
                return solution
            else:
                pass

        return None


# TODO: Add some operator fusions here.
class FusionOptimizer(object):
    def find(self, node_list):
        return None


def _find_an_optimization(node_list):
    optimizers = (FusionOptimizer(), RedundantOptimizer())

    for optm in optimizers:
        solution = optm.find(node_list)
        if solution is not None:
            return solution

    return None


def _apply_optimization(solution, node_list):
    return solution.apply(node_list)


def _build_onnx_model(node_list):
    regenerated = []
    for n_ in node_list:
        regenerated.append(n_.generate())
    return regenerated


def optimize_onnx(onnx_nodes):
    """
    Optimize onnx model by several approaches.
    :param onnx_nodes: the onnx node list in onnx model.
    :return:
    """
    node_list = LinkedNode.build_from_onnx(onnx_nodes)
    solution = _find_an_optimization(node_list)
    while solution:
        node_list = _apply_optimization(solution, node_list)
        solution = _find_an_optimization(node_list)

    return _build_onnx_model(node_list)
