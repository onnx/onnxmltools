try:
    from collections.abc import Mapping, MutableMapping
except:
    from collections import Mapping, MutableMapping
from collections import OrderedDict


class CaseInsensitiveDict(MutableMapping):
    def __init__(self, data=None, **kwargs):
        self._dict = OrderedDict()
        if data:
            self.update(data, **kwargs)

    def __setitem__(self, key, value):
        self._dict[key.casefold()] = (key, value)

    def __getitem__(self, key):
        return self._dict[key.casefold()][1]

    def __delitem__(self, key):
        del self._dict[key.casefold()]

    def __iter__(self):
        return (key for key, _ in self._dict.values())

    def __len__(self):
        return len(self._dict)

    def casefolded(self):
        """Like iteritems(), but casefolded."""
        return (
            (casefolded_key, keyval[1])
            for casefolded_key, keyval
            in self._dict.items()
        )

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        return dict(self.casefolded()) == dict(other.casefolded())

    def copy(self):
         return CaseInsensitiveDict(self._dict.values())

    def __repr__(self):
        return str(dict(self.items()))
