import warnings
from ..convert.common.case_insensitive_dict import CaseInsensitiveDict
from ..proto import onnx, onnx_proto
from distutils.version import StrictVersion


KNOWN_METADATA_PROPS = CaseInsensitiveDict({
    'Image.BitmapPixelFormat': ['gray8', 'rgb8', 'bgr8', 'rgba8', 'bgra8'],
    'Image.ColorSpaceGamma': ['linear', 'srgb'],
    'Image.NominalPixelRange': ['nominalrange_0_255', 'normalized_0_1', 'normalized_1_1', 'nominalrange_16_235'],
})


def _validate_metadata(metadata_props):
    '''
    Validate metadata properties and possibly show warnings or throw exceptions.

    :param metadata_props: A dictionary of metadata properties, with property names and values
    '''
    if len(CaseInsensitiveDict(metadata_props)) != len(metadata_props):
        raise RuntimeError('Duplicate metadata props found')

    for key, value in metadata_props.items():
        valid_values = KNOWN_METADATA_PROPS.get(key)
        if valid_values and value.lower() not in valid_values:
            warnings.warn('Key {} has invalid value {}. Valid values are {}'.format(key, value, valid_values))


def add_metadata_props(onnx_model, metadata_props, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        warnings.warn('Metadata properties are not supported in targeted ONNX-%s' % targeted_onnx)
        return
    _validate_metadata(metadata_props)
    new_metadata = CaseInsensitiveDict({x.key: x.value for x in onnx_model.metadata_props})
    new_metadata.update(metadata_props)
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(
        onnx_proto.StringStringEntryProto(key=key, value=value)
        for key, value in metadata_props.items()
    )


def set_denotation(onnx_model, input_name, denotation, dimension_denotation=None, targeted_onnx=onnx.__version__):
    if StrictVersion(targeted_onnx) < StrictVersion('1.2.1'):
        warnings.warn('Denotation is not supported in targeted ONNX-%s' % targeted_onnx)
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
