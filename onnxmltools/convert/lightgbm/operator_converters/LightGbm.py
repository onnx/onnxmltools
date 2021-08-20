# SPDX-License-Identifier: Apache-2.0

import copy
import numbers
from collections import deque, Counter
import ctypes
import json
import numpy as np
from ...common._apply_operation import (
    apply_div, apply_reshape, apply_sub, apply_cast, apply_identity, apply_clip)
from ...common._registration import register_converter
from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ....proto import onnx_proto


def has_tqdm():
    try:
        from tqdm import tqdm  # noqa
        return True
    except ImportError:
        return False


def _translate_split_criterion(criterion):
    # If the criterion is true, LightGBM use the left child.
    # Otherwise, right child is selected.
    if criterion == '<=':
        return 'BRANCH_LEQ'
    elif criterion == '<':
        return 'BRANCH_LT'
    elif criterion == '>=':
        return 'BRANCH_GTE'
    elif criterion == '>':
        return 'BRANCH_GT'
    elif criterion == '==':
        return 'BRANCH_EQ'
    elif criterion == '!=':
        return 'BRANCH_NEQ'
    else:
        raise ValueError(
            'Unsupported splitting criterion: %s. Only <=, '
            '<, >=, and > are allowed.')


def _create_node_id(node_id_pool):
    i = 0
    while i in node_id_pool:
        i += 1
    node_id_pool.add(i)
    return i


def _parse_tree_structure(tree_id, class_id, learning_rate,
                          tree_structure, attrs):
    """
    The pool of all nodes' indexes created when parsing a single tree.
    Different tree use different pools.
    """
    node_id_pool = set()
    node_pyid_pool = dict()

    node_id = _create_node_id(node_id_pool)
    node_pyid_pool[id(tree_structure)] = node_id

    # The root node is a leaf node.
    if ('left_child' not in tree_structure or
            'right_child' not in tree_structure):
        _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                    learning_rate, tree_structure, attrs)
        return

    left_pyid = id(tree_structure['left_child'])
    right_pyid = id(tree_structure['right_child'])

    if left_pyid in node_pyid_pool:
        left_id = node_pyid_pool[left_pyid]
        left_parse = False
    else:
        left_id = _create_node_id(node_id_pool)
        node_pyid_pool[left_pyid] = left_id
        left_parse = True

    if right_pyid in node_pyid_pool:
        right_id = node_pyid_pool[right_pyid]
        right_parse = False
    else:
        right_id = _create_node_id(node_id_pool)
        node_pyid_pool[right_pyid] = right_id
        right_parse = True

    attrs['nodes_treeids'].append(tree_id)
    attrs['nodes_nodeids'].append(node_id)

    attrs['nodes_featureids'].append(tree_structure['split_feature'])
    attrs['nodes_modes'].append(
        _translate_split_criterion(tree_structure['decision_type']))
    if isinstance(tree_structure['threshold'], str):
        try:
            attrs['nodes_values'].append(float(tree_structure['threshold']))
        except ValueError:
            import pprint
            text = pprint.pformat(tree_structure)
            if len(text) > 100000:
                text = text[:100000] + "\n..."
            raise TypeError("threshold must be a number not '{}'"
                            "\n{}".format(tree_structure['threshold'], text))
    else:
        attrs['nodes_values'].append(tree_structure['threshold'])

    # Assume left is the true branch and right is the false branch
    attrs['nodes_truenodeids'].append(left_id)
    attrs['nodes_falsenodeids'].append(right_id)
    if tree_structure['default_left']:
        if tree_structure["missing_type"] == 'None' and float(tree_structure['threshold']) < 0.0:
            attrs['nodes_missing_value_tracks_true'].append(0)
        else:
            attrs['nodes_missing_value_tracks_true'].append(1)
    else:
        attrs['nodes_missing_value_tracks_true'].append(0)
    attrs['nodes_hitrates'].append(1.)
    if left_parse:
        _parse_node(
            tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['left_child'], attrs)
    if right_parse:
        _parse_node(
            tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
            learning_rate, tree_structure['right_child'], attrs)


def _parse_node(tree_id, class_id, node_id, node_id_pool, node_pyid_pool,
                learning_rate, node, attrs):
    """
    Parses nodes.
    """
    if ((hasattr(node, 'left_child') and hasattr(node, 'right_child')) or
            ('left_child' in node and 'right_child' in node)):

        left_pyid = id(node['left_child'])
        right_pyid = id(node['right_child'])

        if left_pyid in node_pyid_pool:
            left_id = node_pyid_pool[left_pyid]
            left_parse = False
        else:
            left_id = _create_node_id(node_id_pool)
            node_pyid_pool[left_pyid] = left_id
            left_parse = True

        if right_pyid in node_pyid_pool:
            right_id = node_pyid_pool[right_pyid]
            right_parse = False
        else:
            right_id = _create_node_id(node_id_pool)
            node_pyid_pool[right_pyid] = right_id
            right_parse = True

        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)

        attrs['nodes_featureids'].append(node['split_feature'])
        attrs['nodes_modes'].append(
            _translate_split_criterion(node['decision_type']))
        if isinstance(node['threshold'], str):
            try:
                attrs['nodes_values'].append(float(node['threshold']))
            except ValueError:
                import pprint
                text = pprint.pformat(node)
                if len(text) > 100000:
                    text = text[:100000] + "\n..."
                raise TypeError("threshold must be a number not '{}'"
                                "\n{}".format(node['threshold'], text))
        else:
            attrs['nodes_values'].append(node['threshold'])

        # Assume left is the true branch
        # and right is the false branch
        attrs['nodes_truenodeids'].append(left_id)
        attrs['nodes_falsenodeids'].append(right_id)
        if node['default_left']:
            if node['missing_type'] == 'None' and float(node['threshold']) < 0.0:
                attrs['nodes_missing_value_tracks_true'].append(0)
            else:
                attrs['nodes_missing_value_tracks_true'].append(1)
        else:
            attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Recursively dive into the child nodes
        if left_parse:
            _parse_node(
                tree_id, class_id, left_id, node_id_pool, node_pyid_pool,
                learning_rate, node['left_child'], attrs)
        if right_parse:
            _parse_node(
                tree_id, class_id, right_id, node_id_pool, node_pyid_pool,
                learning_rate, node['right_child'], attrs)
    elif hasattr(node, 'left_child') or hasattr(node, 'right_child'):
        raise ValueError('Need two branches')
    else:
        # Node attributes
        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_featureids'].append(0)
        attrs['nodes_modes'].append('LEAF')
        # Leaf node has no threshold.
        # A zero is appended but it will never be used.
        attrs['nodes_values'].append(0.)
        # Leaf node has no child.
        # A zero is appended but it will never be used.
        attrs['nodes_truenodeids'].append(0)
        # Leaf node has no child.
        # A zero is appended but it will never be used.
        attrs['nodes_falsenodeids'].append(0)
        # Leaf node has no split function.
        # A zero is appended but it will never be used.
        attrs['nodes_missing_value_tracks_true'].append(0)
        attrs['nodes_hitrates'].append(1.)

        # Leaf attributes
        attrs['class_treeids'].append(tree_id)
        attrs['class_nodeids'].append(node_id)
        attrs['class_ids'].append(class_id)
        attrs['class_weights'].append(
            float(node['leaf_value']) * learning_rate)


def dump_booster_model(self, num_iteration=None, start_iteration=0,
                       importance_type='split', verbose=0):
    """
    Dumps Booster to JSON format.

    Parameters
    ----------
    self: booster
    num_iteration : int or None, optional (default=None)
        Index of the iteration that should be dumped.
        If None, if the best iteration exists, it is dumped; otherwise,
        all iterations are dumped.
        If <= 0, all iterations are dumped.
    start_iteration : int, optional (default=0)
        Start index of the iteration that should be dumped.
    importance_type : string, optional (default="split")
        What type of feature importance should be dumped.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    verbose: dispays progress (usefull for big trees)

    Returns
    -------
    json_repr : dict
        JSON format of Booster.

    .. note::
        This function is inspired from
        the *lightgbm* (`dump_model
        <https://lightgbm.readthedocs.io/en/latest/pythonapi/
        lightgbm.Booster.html#lightgbm.Booster.dump_model>`_.
        It creates intermediate structure to speed up the conversion
        into ONNX of such model. The function overwrites the
        `json.load` to fastly extract nodes.
    """
    if getattr(self, 'is_mock', False):
        return self.dump_model(), None
    from lightgbm.basic import (
        _LIB, FEATURE_IMPORTANCE_TYPE_MAPPER, _safe_call,
        json_default_with_numpy)
    if num_iteration is None:
        num_iteration = self.best_iteration
    importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
    buffer_len = 1 << 20
    tmp_out_len = ctypes.c_int64(0)
    string_buffer = ctypes.create_string_buffer(buffer_len)
    ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
    if verbose >= 2:
        print("[dump_booster_model] call CAPI: LGBM_BoosterDumpModel")
    _safe_call(_LIB.LGBM_BoosterDumpModel(
        self.handle,
        ctypes.c_int(start_iteration),
        ctypes.c_int(num_iteration),
        ctypes.c_int(importance_type_int),
        ctypes.c_int64(buffer_len),
        ctypes.byref(tmp_out_len),
        ptr_string_buffer))
    actual_len = tmp_out_len.value
    # if buffer length is not long enough, reallocate a buffer
    if actual_len > buffer_len:
        string_buffer = ctypes.create_string_buffer(actual_len)
        ptr_string_buffer = ctypes.c_char_p(
            *[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int(importance_type_int),
            ctypes.c_int64(actual_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))

    class Hook(json.JSONDecoder):
        """
        Keep track of the progress, stores a copy of all objects with
        a decision into a different container in order to walk through
        all nodes in a much faster way than going through the architecture.
        """
        def __init__(self, *args, info=None, n_trees=None, verbose=0,
                     **kwargs):
            json.JSONDecoder.__init__(
                self, object_hook=self.hook, *args, **kwargs)
            self.nodes = []
            self.buffer = []
            self.info = info
            self.n_trees = n_trees
            self.verbose = verbose
            self.stored = 0
            if verbose >= 2 and n_trees is not None and has_tqdm():
                from tqdm import tqdm
                self.loop = tqdm(total=n_trees)
                self.loop.set_description("dump_booster")
            else:
                self.loop = None

        def hook(self, obj):
            """
            Hook called everytime a JSON object is created.
            Keep track of the progress, stores a copy of all objects with
            a decision into a different container.
            """
            # Every obj goes through this function from the leaves to the root.
            if 'tree_info' in obj:
                self.info['decision_nodes'] = self.nodes
                if self.n_trees is not None and len(self.nodes) != self.n_trees:
                    raise RuntimeError(
                        "Unexpected number of trees %d (expecting %d)." % (
                            len(self.nodes), self.n_trees))
                self.nodes = []
                if self.loop is not None:
                    self.loop.close()
            if 'tree_structure' in obj:
                self.nodes.append(self.buffer)
                if self.loop is not None:
                    self.loop.update(len(self.nodes))
                    if len(self.nodes) % 10 == 0:
                        self.loop.set_description(
                            "dump_booster: %d/%d trees, %d nodes" % (
                                len(self.nodes), self.n_trees, self.stored))
                self.buffer = []
            if "decision_type" in obj:
                self.buffer.append(obj)
                self.stored += 1
            return obj

    if verbose >= 2:
        print("[dump_booster_model] to_json")
    info = {}
    ret = json.loads(string_buffer.value.decode('utf-8'), cls=Hook,
                     info=info, n_trees=self.num_trees(), verbose=verbose)
    ret['pandas_categorical'] = json.loads(
        json.dumps(self.pandas_categorical,
                   default=json_default_with_numpy))
    if verbose >= 2:
        print("[dump_booster_model] end.")
    return ret, info


def convert_lightgbm(scope, operator, container):
    """
    Converters for *lightgbm*.
    """
    verbose = getattr(container, 'verbose', 0)
    gbm_model = operator.raw_operator
    gbm_text, info = dump_booster_model(gbm_model.booster_, verbose=verbose)
    modify_tree_for_rule_in_set(gbm_text, use_float=True, verbose=verbose, info=info)

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = operator.full_name

    # Create different attributes for classifier and
    # regressor, respectively
    post_transform = None
    if gbm_text['objective'].startswith('binary'):
        n_classes = 1
        attrs['post_transform'] = 'LOGISTIC'
    elif gbm_text['objective'].startswith('multiclass'):
        n_classes = gbm_text['num_class']
        attrs['post_transform'] = 'SOFTMAX'
    elif gbm_text['objective'].startswith('regression'):
        n_classes = 1  # Regressor has only one output variable
        attrs['post_transform'] = 'NONE'
        attrs['n_targets'] = n_classes
    elif gbm_text['objective'].startswith(('poisson', 'gamma')):
        n_classes = 1  # Regressor has only one output variable
        attrs['n_targets'] = n_classes
        # 'Exp' is not a supported post_transform value in the ONNX spec yet,
        # so we need to add an 'Exp' post transform node to the model
        attrs['post_transform'] = 'NONE'
        post_transform = "Exp"
    else:
        raise RuntimeError(
            "LightGBM objective should be cleaned already not '{}'.".format(
                gbm_text['objective']))

    # Use the same algorithm to parse the tree
    for i, tree in enumerate(gbm_text['tree_info']):
        tree_id = i
        class_id = tree_id % n_classes
        # tree['shrinkage'] --> LightGbm provides figures with it already.
        learning_rate = 1.
        _parse_tree_structure(
            tree_id, class_id, learning_rate, tree['tree_structure'], attrs)

    # Sort nodes_* attributes. For one tree, its node indexes
    # should appear in an ascent order in nodes_nodeids. Nodes
    # from a tree with a smaller tree index should appear
    # before trees with larger indexes in nodes_nodeids.
    node_numbers_per_tree = Counter(attrs['nodes_treeids'])
    tree_number = len(node_numbers_per_tree.keys())
    accumulated_node_numbers = [0] * tree_number
    for i in range(1, tree_number):
        accumulated_node_numbers[i] = (
            accumulated_node_numbers[i - 1] + node_numbers_per_tree[i - 1])
    global_node_indexes = []
    for i in range(len(attrs['nodes_nodeids'])):
        tree_id = attrs['nodes_treeids'][i]
        node_id = attrs['nodes_nodeids'][i]
        global_node_indexes.append(
            accumulated_node_numbers[tree_id] + node_id)
    for k, v in attrs.items():
        if k.startswith('nodes_'):
            merged_indexes = zip(
                copy.deepcopy(global_node_indexes), v)
            sorted_list = [pair[1]
                           for pair in sorted(merged_indexes,
                                              key=lambda x: x[0])]
            attrs[k] = sorted_list

    # Create ONNX object
    if (gbm_text['objective'].startswith('binary') or
            gbm_text['objective'].startswith('multiclass')):
        # Prepare label information for both of TreeEnsembleClassifier
        class_type = onnx_proto.TensorProto.STRING
        if all(isinstance(i, (numbers.Real, bool, np.bool_))
               for i in gbm_model.classes_):
            class_type = onnx_proto.TensorProto.INT64
            class_labels = [int(i) for i in gbm_model.classes_]
            attrs['classlabels_int64s'] = class_labels
        elif all(isinstance(i, str) for i in gbm_model.classes_):
            class_labels = [str(i) for i in gbm_model.classes_]
            attrs['classlabels_strings'] = class_labels
        else:
            raise ValueError(
                'Only string and integer class labels are allowed')

        # Create tree classifier
        probability_tensor_name = scope.get_unique_variable_name(
            'probability_tensor')
        label_tensor_name = scope.get_unique_variable_name('label_tensor')

        container.add_node(
            'TreeEnsembleClassifier', operator.input_full_names,
            [label_tensor_name, probability_tensor_name],
            op_domain='ai.onnx.ml', **attrs)

        prob_tensor = probability_tensor_name

        if gbm_model.boosting_type == 'rf':
            col_index_name = scope.get_unique_variable_name('col_index')
            first_col_name = scope.get_unique_variable_name('first_col')
            zeroth_col_name = scope.get_unique_variable_name('zeroth_col')
            denominator_name = scope.get_unique_variable_name('denominator')
            modified_first_col_name = scope.get_unique_variable_name(
                'modified_first_col')
            unit_float_tensor_name = scope.get_unique_variable_name(
                'unit_float_tensor')
            merged_prob_name = scope.get_unique_variable_name('merged_prob')
            predicted_label_name = scope.get_unique_variable_name(
                'predicted_label')
            classes_name = scope.get_unique_variable_name('classes')
            final_label_name = scope.get_unique_variable_name('final_label')

            container.add_initializer(
                col_index_name, onnx_proto.TensorProto.INT64, [], [1])
            container.add_initializer(
                unit_float_tensor_name, onnx_proto.TensorProto.FLOAT,
                [], [1.0])
            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [],
                [100.0])
            container.add_initializer(classes_name, class_type,
                                      [len(class_labels)], class_labels)

            container.add_node(
                'ArrayFeatureExtractor',
                [probability_tensor_name, col_index_name],
                first_col_name,
                name=scope.get_unique_operator_name(
                    'ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_div(scope, [first_col_name, denominator_name],
                      modified_first_col_name, container, broadcast=1)
            apply_sub(
                scope, [unit_float_tensor_name, modified_first_col_name],
                zeroth_col_name, container, broadcast=1)
            container.add_node(
                'Concat', [zeroth_col_name, modified_first_col_name],
                merged_prob_name,
                name=scope.get_unique_operator_name('Concat'), axis=1)
            container.add_node(
                'ArgMax', merged_prob_name,
                predicted_label_name,
                name=scope.get_unique_operator_name('ArgMax'), axis=1)
            container.add_node(
                'ArrayFeatureExtractor', [classes_name, predicted_label_name],
                final_label_name,
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            apply_reshape(scope, final_label_name,
                          operator.outputs[0].full_name,
                          container, desired_shape=[-1, ])
            prob_tensor = merged_prob_name
        else:
            container.add_node('Identity', label_tensor_name,
                               operator.outputs[0].full_name,
                               name=scope.get_unique_operator_name('Identity'))

        # Convert probability tensor to probability map
        # (keys are labels while values are the associated probabilities)
        container.add_node('Identity', prob_tensor,
                           operator.outputs[1].full_name)
    else:
        # Create tree regressor
        output_name = scope.get_unique_variable_name('output')

        keys_to_be_renamed = list(
            k for k in attrs if k.startswith('class_'))

        for k in keys_to_be_renamed:
            # Rename class_* attribute to target_*
            # because TreeEnsebmleClassifier
            # and TreeEnsembleClassifier have different ONNX attributes
            attrs['target' + k[5:]] = copy.deepcopy(attrs[k])
            del attrs[k]
        container.add_node(
            'TreeEnsembleRegressor', operator.input_full_names,
            output_name, op_domain='ai.onnx.ml', **attrs)
        if gbm_model.boosting_type == 'rf':
            denominator_name = scope.get_unique_variable_name('denominator')

            container.add_initializer(
                denominator_name, onnx_proto.TensorProto.FLOAT, [], [100.0])

            apply_div(scope, [output_name, denominator_name],
                      operator.output_full_names, container, broadcast=1)
        elif post_transform:
            container.add_node(
                post_transform,
                output_name,
                operator.output_full_names,
                name=scope.get_unique_operator_name(
                    post_transform),
            )
        else:
            container.add_node('Identity', output_name,
                               operator.output_full_names,
                               name=scope.get_unique_operator_name('Identity'))


def modify_tree_for_rule_in_set(gbm, use_float=False, verbose=0, count=0,  # pylint: disable=R1710
                                info=None):
    """
    LightGBM produces sometimes a tree with a node set
    to use rule ``==`` to a set of values (= in set),
    the values are separated by ``||``.
    This function unfold theses nodes.

    :param gbm: a tree coming from lightgbm dump
    :param use_float: use float otherwise int first
        then float if it does not work
    :param verbose: verbosity, use *tqdm* to show progress
    :param count: number of nodes already changed (origin) before this call
    :param info: addition information to speed up this search
    :return: number of changed nodes (include *count*)
    """
    if 'tree_info' in gbm:
        if info is not None:
            dec_nodes = info['decision_nodes']
        else:
            dec_nodes = None
        if verbose >= 2 and has_tqdm():
            from tqdm import tqdm
            loop = tqdm(gbm['tree_info'])
            for i, tree in enumerate(loop):
                loop.set_description("rules tree %d c=%d" % (i, count))
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if dec_nodes is None else dec_nodes[i])
        else:
            for i, tree in enumerate(gbm['tree_info']):
                count = modify_tree_for_rule_in_set(
                    tree, use_float=use_float, count=count,
                    info=None if dec_nodes is None else dec_nodes[i])
        return count

    if 'tree_structure' in gbm:
        return modify_tree_for_rule_in_set(
            gbm['tree_structure'], use_float=use_float, count=count,
            info=info)

    if 'decision_type' not in gbm:
        return count

    def str2number(val):
        if use_float:
            return float(val)
        else:
            try:
                return int(val)
            except ValueError:  # pragma: no cover
                return float(val)

    if info is None:

        def recursive_call(this, c):
            if 'left_child' in this:
                c = process_node(this['left_child'], count=c)
            if 'right_child' in this:
                c = process_node(this['right_child'], count=c)
            return c

        def process_node(node, count):
            if 'decision_type' not in node:
                return count
            if node['decision_type'] != '==':
                return recursive_call(node, count)
            th = node['threshold']
            if not isinstance(th, str):
                return recursive_call(node, count)
            pos = th.find('||')
            if pos == -1:
                return recursive_call(node, count)
            th1 = str2number(th[:pos])

            def doit():
                rest = th[pos + 2:]
                if '||' not in rest:
                    rest = str2number(rest)

                node['threshold'] = th1
                new_node = node.copy()
                node['right_child'] = new_node
                new_node['threshold'] = rest

            doit()
            return recursive_call(node, count + 1)

        return process_node(gbm, count)

    # when info is used

    def split_node(node, th, pos):
        th1 = str2number(th[:pos])

        rest = th[pos + 2:]
        if '||' not in rest:
            rest = str2number(rest)
            app = False
        else:
            app = True

        node['threshold'] = th1
        new_node = node.copy()
        node['right_child'] = new_node
        new_node['threshold'] = rest
        return new_node, app

    stack = deque(info)
    while len(stack) > 0:
        node = stack.pop()

        if 'decision_type' not in node:
            continue  # leave

        if node['decision_type'] != '==':
            continue

        th = node['threshold']
        if not isinstance(th, str):
            continue

        pos = th.find('||')
        if pos == -1:
            continue

        new_node, app = split_node(node, th, pos)
        count += 1
        if app:
            stack.append(new_node)

    return count


def convert_lgbm_zipmap(scope, operator, container):
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    if hasattr(operator, 'classlabels_int64s'):
        zipmap_attrs['classlabels_int64s'] = operator.classlabels_int64s
        to_type = onnx_proto.TensorProto.INT64
    elif hasattr(operator, 'classlabels_strings'):
        zipmap_attrs['classlabels_strings'] = operator.classlabels_strings
        to_type = onnx_proto.TensorProto.STRING
    else:
        raise RuntimeError("Unknown class type.")
    if to_type == onnx_proto.TensorProto.STRING:
        apply_identity(scope, operator.inputs[0].full_name,
                       operator.outputs[0].full_name, container)
    else:
        apply_cast(scope, operator.inputs[0].full_name,
                   operator.outputs[0].full_name, container, to=to_type)

    if operator.zipmap:
        container.add_node('ZipMap', operator.inputs[1].full_name,
                           operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)
    else:
        # This should be apply_identity but optimization fails in
        # onnxconverter-common when trying to remove identity nodes.
        apply_clip(scope, operator.inputs[1].full_name,
                   operator.outputs[1].full_name, container,
                   min=np.array([0], dtype=np.float32),
                   max=np.array([1], dtype=np.float32))


register_converter('LgbmClassifier', convert_lightgbm)
register_converter('LgbmRegressor', convert_lightgbm)
register_converter('LgbmZipMap', convert_lgbm_zipmap)
