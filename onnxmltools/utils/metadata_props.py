from ..convert.common.case_insensitive_dict import CaseInsensitiveDict
from ..proto import onnx, onnx_proto
from distutils.version import StrictVersion


def _get_case_insensitive(iterable, key):
    return next((x for x in iterable if x.lower() == key.lower()), None)


def _validate_metadata(metadata_props):
    valid_image_metadata_props = {
        'Image.BitmapPixelFormat': ['Gray8', 'Rgb8', 'Bgr8', 'Rgba8', 'Bgra8'],
        'Image.ColorSpaceGamma': ['Linear', 'SRGB'],
        'Image.NominalPixelRange': ['NominalRange_0_255', 'Normalized_0_1', 'Normalized_1_1', 'NominalRange_16_235'],
    }
    case_insensitive_metadata_props = CaseInsensitiveDict(metadata_props)
    if len(case_insensitive_metadata_props) != len(metadata_props):
        raise RuntimeError('Duplicate metadata props found')

    for key, value in metadata_props.items():
        valid_values = valid_image_metadata_props.pop(key)
        if valid_values and value.casefold() not in (x.casefold() for x in valid_values):
            print('Warning: value {} is invalid. Valid values are {}'.format(value, valid_values))

    if 0 < len(valid_image_metadata_props) < 3:
        print('Warning: incomplete image metadata is being added. Keys {} are missing.'.format(', '.join(valid_image_metadata_props)))


def add_metadata_props(onnx_model, metadata_props, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        raise RuntimeError('Metadata properties are not supported in targeted ONNX-%s' % targeted_onnx)
    _validate_metadata(metadata_props)
    new_metadata = CaseInsensitiveDict({x.key: x.value for x in onnx_model.metadata_props})
    new_metadata.update(metadata_props)
    model_metadata = [
        onnx_proto.StringStringEntryProto(key=key, value=value)
        for key, value in metadata_props.items()
    ]


def set_denotation(onnx_model, input_name, denotation, dimension_denotation=None, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        raise RuntimeError('Metadata properties are not supported in targeted ONNX-%s' % targeted_onnx)
    for graph_input in onnx_model.graph.input:
        if graph_input.name == input_name:
            graph_input.type.denotation = denotation
            if dimension_denotation:
                dimensions = graph_input.type.tensor_type.shape.dim
                if len(dimension_denotation) != len(dimensions):
                    raise RuntimeError('Wrong number of dimensions: input "{}" has {} dimensions'.format(input_name, len(dimensions)))
                for dimension, channel_denotation in zip(dimensions, dimension_denotation):
                    dimension.denotation = channel_denotation
            return onnx_model
    raise RuntimeError('Input "{}" not found'.format(input_name))
