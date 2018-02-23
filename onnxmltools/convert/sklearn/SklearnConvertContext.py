#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import ExtendedConvertContext


class SklearnConvertContext(ExtendedConvertContext):
    '''
    The ConvertContext provides data about the conversion,
    specifically keeping a mapping of the old->new names as 
    well as provides helper functions for generating unique names.
    '''
    pass