from ..proto import onnx, onnx_proto
from distutils.version import StrictVersion


def _get_case_insensitive(iterable, key):
    return next((x for x in iterable if x.lower() == key.lower()), None)


def _validate_metadata(metadata_props):
    valid_metadata_props = {
        'Image.BitmapPixelFormat': ['Gray8', 'Rgb8', 'Bgr8', 'Rgba8', 'Bgra8'],
        'Image.ColorSpaceGamma': ['Linear', 'SRGB'],
        'Image.NominalPixelRange': ['NominalRange_0_255', 'Normalized_0_1', 'Normalized_1_1', 'NominalRange_16_235'],
    }
    image_metadata_props = {k: v for k, v in metadata_props.items() if _get_case_insensitive(valid_metadata_props, k)}
    for key, value in image_metadata_props.items():
        key = _get_case_insensitive(valid_metadata_props, key)
        if not key:
            print('Warning: key {} is defined multiple times'.format(key))
        else:
            valid_values = valid_metadata_props.pop(key)
            if not _get_case_insensitive(valid_values, value):
                print('Warning: value {} is invalid. Valid values are {}'.format(value, valid_values))
    if image_metadata_props and valid_metadata_props:
        print('Warning: incomplete image metadata is being added. Keys {} are missing.'.format(', '.join(valid_metadata_props)))

COLOR_SPACE_TO_PIXEL_FORMAT = {
    'BGR': 'Bgr8',
    'RGB': 'Rgb8',
    'GRAY': 'Gray8',
}


def color_space_to_pixel_format(color_space):
    return COLOR_SPACE_TO_PIXEL_FORMAT[color_space]


def add_metadata_props(onnx_model, metadata_props, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        print('Metadata properties are not supported in targeted ONNX-%s' % targeted_onnx)
        return
    _validate_metadata(metadata_props)
    # Overwrite old properties (case insensitive)
    new_props = [x.lower() for x in metadata_props]
    model_metadata = onnx_model.metadata_props
    for prop in model_metadata:
        if prop.key.lower() in new_props:
            model_metadata.remove(prop)
    model_metadata.extend(onnx_proto.StringStringEntryProto(key=key, value=value)
                                     for key, value in metadata_props.items())


def set_denotation(onnx_model, input_name, denotation, dimension_denotation=None, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        print('Metadata properties are not supported in targeted ONNX-%s' % targeted_onnx)
        return
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
