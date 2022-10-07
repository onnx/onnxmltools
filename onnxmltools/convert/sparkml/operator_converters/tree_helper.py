# SPDX-License-Identifier: Apache-2.0

import numpy as np


class Node:

    _names = [
        "class_ids",
        "class_nodeids",
        "class_treeids",
        "class_weights",
        "classlabels_int64s",
        "nodes_falsenodeids",
        "nodes_featureids",
        "nodes_hitrates",
        "nodes_missing_value_tracks_true",
        "nodes_modes",
        "nodes_nodeids",
        "nodes_treeids",
        "nodes_truenodeids",
        "nodes_values",
    ]

    def __init__(self, **kwargs):
        for att in Node._names:
            setattr(self, att, None)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        ps = ", ".join(map(lambda k: f"{k}={getattr(self, k)!r}", Node._names))
        return f"Node({ps})"

    @property
    def attrs(self):
        return {k: getattr(self, k) for k in Node._names}

    @staticmethod
    def create(attrs):
        nodes = {}
        root = None
        indices = attrs["nodes_nodeids"]
        for n, nid in enumerate(indices):
            tid = attrs["nodes_treeids"][n]
            mode = attrs["nodes_modes"][n]
            kwargs = {}
            for k, v in attrs.items():
                if not k.startswith("nodes"):
                    continue
                kwargs[k] = v[n]
            if mode == "LEAF":
                pos = [
                    i
                    for i, (t, c) in enumerate(
                        zip(attrs["class_treeids"], attrs["class_nodeids"])
                    )
                    if t == tid and c == nid
                ]
                for k, v in attrs.items():
                    if k in {"post_transform", "name", "domain"}:
                        continue
                    if k.startswith("nodes"):
                        continue
                    if "label" in k:
                        kwargs[k] = v
                        continue
                    kwargs[k] = [v[p] for p in pos]

            node = Node(**kwargs)
            if mode == "BRANCH_LEQ" and isinstance(
                node.nodes_values, (np.ndarray, list)
            ):
                node.nodes_modes = "||"
            nodes[tid, nid] = node
            if root is None:
                root = node

        # add links
        for k, v in nodes.items():
            if v.nodes_modes != "LEAF":
                ntrue = nodes[v.nodes_treeids, v.nodes_truenodeids]
                nfalse = nodes[v.nodes_treeids, v.nodes_falsenodeids]
                v._nodes_truenode = ntrue
                v._nodes_falsenode = nfalse
        return root, nodes

    def enumerate_nodes(self, unique=False):
        if unique:
            done = set()
            yield self
            done.add(id(self))
            if self.nodes_modes != "LEAF":
                for node in self._nodes_truenode:
                    if id(node) not in done:
                        yield node
                        done.add(id(node))
                for node in self._nodes_falsenode:
                    if id(node) not in done:
                        yield node
                        done.add(id(node))
        else:
            yield self
            if self.nodes_modes != "LEAF":
                for node in self._nodes_truenode:
                    yield node
                for node in self._nodes_falsenode:
                    yield node

    def __iter__(self):
        "Iterates over the node."
        for node in self.enumerate_nodes(unique=True):
            yield node

    def _unfold_rule_or(self):
        if self.nodes_modes == "||":
            values = self.nodes_values
            if len(values) == 1:
                self.nodes_modes = "BRANCH_EQ"
                self.nodes_values = self.nodes_values[0]
                return True
            if len(values) == 0:
                raise ValueError(f"Issue with {self!r}.")
            th = self.nodes_values[-1]
            vals = self.nodes_values[:-1]

            new_node = Node(**self.attrs)
            new_node.nodes_values = th
            new_node.nodes_modes = "BRANCH_EQ"
            new_node._nodes_truenode = self._nodes_truenode
            new_node._nodes_falsenode = self._nodes_falsenode

            self.nodes_values = vals
            self._nodes_falsenode = new_node
            return True
        return False

    def unfold_rule_or(self):
        r = True
        while r:
            r = False
            for node in self:
                r = node._unfold_rule_or()
                if r:
                    break
        self.set_new_numbers()

    def set_new_numbers(self):
        # new number
        tree = {}
        done = set()
        stack = [self]
        while len(stack) > 0:
            node = stack[0]
            stack = stack[1:]
            if id(node) in done:
                continue
            done.add(id(node))
            tid = node.nodes_treeids
            if tid not in tree:
                tree[tid] = -1
            nid = tree[tid] + 1
            tree[tid] += 1
            node.nodes_nodeids = nid
            if node.nodes_modes != "LEAF":
                stack.append(node._nodes_truenode)
                stack.append(node._nodes_falsenode)
        # all ids
        ids = {}
        for node in self:
            ids[id(node)] = node.nodes_nodeids
        # replaces true, false ids
        for node in self:
            if node.nodes_modes == "LEAF":
                continue
            node.nodes_falsenodeids = ids[id(node._nodes_falsenode)]
            node.nodes_truenodeids = ids[id(node._nodes_truenode)]

    def to_attrs(self, **kwargs):
        """
        Returns the nodes as node attributes for ONNX.
        """
        nodes = {}
        for node in self:
            nodes[node.nodes_treeids, node.nodes_nodeids] = node
        sort = [v for k, v in sorted(nodes.items())]
        attrs = {}
        for name in Node._names:
            attrs[name] = []
        for node in sort:
            for k in Node._names:
                if not k.startswith("nodes"):
                    continue
                attrs[k].append(getattr(node, k))
            if node.nodes_modes == "LEAF":
                for k in Node._names:
                    if k.startswith("nodes"):
                        continue
                    if k not in attrs:
                        attrs[k] = []
                    if k == "class_nodeids":
                        attrs[k].extend([node.nodes_nodeids for k in node.class_ids])
                    else:
                        attrs[k].extend(getattr(node, k))
        attrs.update(kwargs)
        return attrs
