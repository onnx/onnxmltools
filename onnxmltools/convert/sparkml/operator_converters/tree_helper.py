# SPDX-License-Identifier: Apache-2.0

import numpy as np


class Node:
    _names_classifier = [
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

    _names_regressor = [
        "target_ids",
        "target_nodeids",
        "target_treeids",
        "target_weights",
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

    def __init__(self, is_classifier, **kwargs):
        if is_classifier:
            self.is_classifier = True
            self._names = Node._names_classifier
        else:
            self.is_classifier = False
            self._names = Node._names_regressor
        for att in self._names:
            setattr(self, att, None)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        ps = ", ".join(map(lambda k: f"{k}={getattr(self, k)!r}", self._names))
        return f"Node({ps})"

    @property
    def attrs(self):
        return {k: getattr(self, k) for k in self._names}

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
                if "class_treeids" in attrs:
                    # classifier
                    pos = [
                        i
                        for i, (t, c) in enumerate(
                            zip(attrs["class_treeids"], attrs["class_nodeids"])
                        )
                        if t == tid and c == nid
                    ]
                else:
                    # regressor
                    pos = [
                        i
                        for i, (t, c) in enumerate(
                            zip(attrs["target_treeids"], attrs["target_nodeids"])
                        )
                        if t == tid and c == nid
                    ]
                for k, v in attrs.items():
                    if k in {"post_transform", "name", "domain", "n_targets"}:
                        continue
                    if k.startswith("nodes"):
                        continue
                    if "label" in k:
                        kwargs[k] = v
                        continue
                    try:
                        kwargs[k] = [v[p] for p in pos]
                    except TypeError as e:
                        raise TypeError(f"Unabel to update attribute {k!r}.") from e

            node = Node("class_treeids" in attrs, **kwargs)
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
                return self
            if len(values) == 0:
                raise ValueError(f"Issue with {self!r}.")
            th = self.nodes_values[-1]
            vals = self.nodes_values[:-1]

            new_node = Node(self.is_classifier, **self.attrs)
            new_node.nodes_values = th
            new_node.nodes_modes = "BRANCH_EQ"
            new_node._nodes_truenode = self._nodes_truenode
            new_node._nodes_falsenode = self._nodes_falsenode

            self.nodes_values = vals
            self._nodes_falsenode = new_node
            return self
        return None

    def unfold_rule_or(self):
        nodes = []
        for node in self:
            if node.nodes_modes == "||":
                nodes.append(node)
        if len(nodes) == 0:
            return
        for node in nodes:
            r = node._unfold_rule_or()
            if r is not None:
                while r is not None:
                    r = r._unfold_rule_or()
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
        for name in self._names:
            attrs[name] = []
        for node in sort:
            for k in self._names:
                if not k.startswith("nodes"):
                    continue
                attrs[k].append(getattr(node, k))
            if node.nodes_modes == "LEAF":
                for k in self._names:
                    if k.startswith("nodes"):
                        continue
                    if k not in attrs:
                        attrs[k] = []
                    if k == "class_nodeids":
                        attrs[k].extend([node.nodes_nodeids for _ in node.class_ids])
                    else:
                        try:
                            attrs[k].extend(getattr(node, k))
                        except TypeError as e:
                            raise TypeError(f"Issue with attribute {k!r}.") from e
        attrs.update(kwargs)

        # update numbers
        new_numbers = {}
        for tid, nid, md in sorted(
            zip(attrs["nodes_treeids"], attrs["nodes_nodeids"], attrs["nodes_modes"])
        ):
            new_numbers[tid, nid] = len(new_numbers)
        for k in [
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_nodeids",
            "class_nodeids",
            "target_nodeids",
        ]:
            if k not in attrs:
                continue
            if "class_" in k or "target_" in k:
                field = k.split("_")[0] + "_treeids"
            else:
                field = "nodes_treeids"
            for i in range(len(attrs[k])):
                nid = attrs[k][i]
                if nid == 0 and k in {"nodes_truenodeids", "nodes_falsenodeids"}:
                    continue
                tid = attrs[field][i]
                new_id = new_numbers[tid, nid]
                attrs[k][i] = new_id
        return attrs


def rewrite_ids_and_process(attrs, logger):
    in_sets_rules = []
    for i, value in enumerate(attrs["nodes_values"]):
        if isinstance(value, (np.ndarray, list)):
            in_sets_rules.append(i)

    logger.info(
        "[convert_decision_tree_classifier] in_set_rules has %d elements",
        len(in_sets_rules),
    )
    for i in in_sets_rules:
        attrs["nodes_modes"][i] = "||"
    logger.info("[convert_decision_tree_classifier] Node.create")
    root, _ = Node.create(attrs)
    logger.info("[convert_decision_tree_classifier] unfold_rule_or")
    root.unfold_rule_or()
    logger.info("[convert_decision_tree_classifier] to_attrs")
    if "class_nodeids" in attrs:
        new_attrs = root.to_attrs(
            post_transform=attrs["post_transform"],
            classlabels_int64s=attrs["classlabels_int64s"],
            name=attrs["name"],
        )
    else:
        new_attrs = root.to_attrs(
            post_transform=attrs["post_transform"],
            n_targets=attrs["n_targets"],
            name=attrs["name"],
        )
    if len(attrs["nodes_nodeids"]) > len(new_attrs["nodes_nodeids"]):
        raise RuntimeError(
            f"The replacement fails as there are less nodes in the new tree, "
            f"{len(attrs['nodes_nodeids'])} > {len(new_attrs['nodes_nodeids'])}."
        )
    if set(attrs) != set(new_attrs):
        raise RuntimeError(
            f"Missing key: {list(sorted(attrs))} != {list(sorted(new_attrs))}."
        )
    logger.info(
        "[convert_decision_tree_classifier] n_nodes=%d", len(attrs["nodes_nodeids"])
    )
    logger.info("[convert_decision_tree_classifier] end")
    return new_attrs
