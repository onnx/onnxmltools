# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx
from .common import utils
import warnings
import importlib


def convert_coreml(model, name=None, initial_types=None, doc_string='', target_opset=None,
                   targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.coreml_installed():
        raise RuntimeError('coremltools is not installed. Please install coremltools to use this feature.')

    from .coreml.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_keras(model, name=None, initial_types=None, doc_string='',
                  target_opset=None, targeted_onnx=onnx.__version__,
                  channel_first_inputs=None, custom_conversion_functions=None, custom_shape_calculators=None,
                  default_batch_size=1):
    if not utils.keras2onnx_installed():
        raise RuntimeError('keras2onnx is not installed. Please install it to use this feature.')

    if custom_conversion_functions:
        warnings.warn('custom_conversion_functions is not supported any more. Please set it to None.')

    from keras2onnx import convert_keras as convert
    return convert(model, name, doc_string, target_opset, channel_first_inputs)


def convert_libsvm(model, name=None, initial_types=None, doc_string='', target_opset=None,
                   targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.libsvm_installed():
        raise RuntimeError('libsvm is not installed. Please install libsvm to use this feature.')

    from .libsvm.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_lightgbm(model, name=None, initial_types=None, doc_string='', target_opset=None,
                     targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.lightgbm_installed():
        raise RuntimeError('lightgbm is not installed. Please install lightgbm to use this feature.')

    from .lightgbm.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_sklearn(model, name=None, initial_types=None, doc_string='', target_opset=None,
                    targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.sklearn_installed():
        raise RuntimeError('scikit-learn is not installed. Please install scikit-learn to use this feature.')

    if not utils.skl2onnx_installed():
        raise RuntimeError('skl2onnx is not installed. Please install skl2onnx to use this feature.')

    from skl2onnx.convert import convert_sklearn as convert_skl2onnx
    return convert_skl2onnx(model, name, initial_types, doc_string, target_opset,
                            custom_conversion_functions, custom_shape_calculators)


def convert_sparkml(model, name=None, initial_types=None, doc_string='', target_opset=None,
                    targeted_onnx=onnx.__version__, custom_conversion_functions=None,
                    custom_shape_calculators=None, spark_session=None):
    if not utils.sparkml_installed():
        raise RuntimeError('Spark is not installed. Please install Spark to use this feature.')

    from .sparkml.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators, spark_session)


def convert_xgboost(*args, **kwargs):
    if not utils.xgboost_installed():
        raise RuntimeError('xgboost is not installed. Please install xgboost to use this feature.')

    from .xgboost.convert import convert
    return convert(*args, **kwargs)


def convert_h2o(*args, **kwargs):
    if not utils.h2o_installed():
        raise RuntimeError('h2o is not installed. Please install h2o to use this feature.')

    from .h2o.convert import convert
    return convert(*args, **kwargs)


def _collect_input_nodes(graph, outputs):
    nodes_to_keep = set()
    input_nodes = set()
    node_inputs = [graph.get_tensor_by_name(ts_).op for ts_ in outputs]
    while node_inputs:
        nd_ = node_inputs[0]
        del node_inputs[0]
        if nd_.type in ['Placeholder', "PlaceholderV2", 'PlaceholderWithDefault']:
            input_nodes.add(nd_)
        if nd_ in nodes_to_keep:
            continue

        nodes_to_keep.add(nd_)
        node_inputs.extend(in_.op for in_ in nd_.inputs)

    return input_nodes, nodes_to_keep


def _convert_tf_wrapper(frozen_graph_def,
                        name=None, input_names=None, output_names=None,
                        doc_string='',
                        target_opset=None,
                        channel_first_inputs=None,
                        debug_mode=False, custom_op_conversions=None):
    """
    convert a tensorflow graph def into a ONNX model proto, just like how keras does.
    :param graph_def: the frozen tensorflow graph
    :param name: the converted onnx model internal name
    :param input_names: the inputs name list of the model
    :param output_names: the output name list of the model
    :param doc_string: doc string
    :param target_opset: the targeted onnx model opset
    :param channel_first_inputs: A list of channel first input (not supported yet)
    :param debug_mode: will enable the log and try to convert as much as possible on conversion
    :return an ONNX ModelProto
    """
    import tensorflow as tf
    import tf2onnx

    if target_opset is None:
        target_opset = onnx.defs.onnx_opset_version()

    if not doc_string:
        doc_string = "converted from {}".format(name)

    tf_graph_def = tf2onnx.tfonnx.tf_optimize(input_names, output_names, frozen_graph_def, True)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tf_graph_def, name='')

        if not input_names:
            input_nodes = list(_collect_input_nodes(tf_graph, output_names)[0])
            input_names = [nd_.outputs[0].name for nd_ in input_nodes]
        g = tf2onnx.tfonnx.process_tf_graph(tf_graph,
                                            continue_on_error=debug_mode,
                                            opset=target_opset,
                                            custom_op_handlers=custom_op_conversions,
                                            inputs_as_nchw=channel_first_inputs,
                                            output_names=output_names,
                                            input_names=input_names)

        onnx_graph = tf2onnx.optimizer.optimize_graph(g)
        model_proto = onnx_graph.make_model(doc_string)

    return model_proto


def convert_tensorflow(frozen_graph_def,
                       name=None, input_names=None, output_names=None,
                       doc_string='',
                       target_opset=None,
                       channel_first_inputs=None,
                       debug_mode=False, custom_op_conversions=None):
    try:
        importlib.import_module('tf2onnx')
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError('tf2onnx is not installed, please install it before calling this function.')

    return _convert_tf_wrapper(frozen_graph_def, name, input_names, output_names, doc_string,
                               target_opset, channel_first_inputs, debug_mode, custom_op_conversions)
