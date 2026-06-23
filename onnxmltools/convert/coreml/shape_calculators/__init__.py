# SPDX-License-Identifier: Apache-2.0

# To register shape calculators for Core ML operators, import associated modules here.
from . import neural_network  # noqa: F401
from . import ArrayFeatureExtractor
from . import Classifier
from . import DictVectorizer
from . import FeatureVectorizer
from . import Identity
from . import OneHotEncoder
from . import Regressor
from . import TensorToLabel
from . import TensorToProbabilityMap
