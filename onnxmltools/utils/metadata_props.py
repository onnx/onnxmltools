from ..proto import onnx_proto


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


def add_metadata_props(onnx_model, metadata_props):
    _validate_metadata(metadata_props)
    onnx_model.metadata_props.extend(onnx_proto.StringStringEntryProto(key=key, value=value)
                                     for key, value in metadata_props.items())
