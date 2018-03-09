import coremltools
from uuid import uuid4
from . import _converters
from . import _parser
from ... import __version__
from ...proto import onnx_proto

def convert(model, name=None, initial_types={}, doc_string=''):
    if isinstance(model, coremltools.models.MLModel):
        spec = model.get_spec()
    else:
        spec = model

    if name is None:
        name = str(uuid4().hex)

    topology = _parser.parse_coreml(spec, initial_types)
    #_parser.visualize_topology(topology, filename=name, view=True)
    onnx_model = _converters.convert_topology(topology, name)
    onnx_model.ir_version = onnx_proto.IR_VERSION
    onnx_model.producer_name = 'winmltools'
    onnx_model.producer_version = __version__
    onnx_model.doc_string = doc_string
    return onnx_model


