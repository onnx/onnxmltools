#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

class ConvertContext:
    '''
    The ConvertContext provides data about the conversion, specifically keeping a mapping of the old->new names as
    well as provides helper functions for generating unique names
    '''

    #Named mapping that allows for specifying an override to the name of the node
    _name_override_map = {
        'ArrayFeatureExtractor':'AFE',
        'DictVectorizer':'DV',
        'FeatureVectorizer': 'FV',
        'OneHotEncoder': 'OHE'
    }

    def __init__(self):
        self._unique_name_set = set()
        self.top_level_inputs = []
        self.top_level_outputs = []

    def get_unique_name(self, name):
        return self.__generate_name(name)

    def __generate_name(self, name):
        if name in self._name_override_map:
            _name = self._name_override_map[name]
        else:
            _name = name

        count = 1
        gen_name = _name
        while gen_name in self._unique_name_set:
            gen_name = "{}.{}".format(_name, count)
            count += 1
        self._unique_name_set.add(gen_name)
        return gen_name


class ExtendedConvertContext(ConvertContext):
    '''
    The ConvertContext provides data about the conversion, specifically keeping a mapping of the old->new names as
    well as provides helper functions for generating unique names
    '''

    def __init__(self):
        ConvertContext.__init__(self)
        self._outputs = []

    @property
    def outputs(self):
        return self._outputs

    def add_output(self, output):
        self._outputs.append(output)

    def clear_outputs(self):
        self._outputs.clear()
