# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# To register shape calculators for Core ML operators, import associated modules here.
from . import ArrayFeatureExtractor
from . import Classifier
from . import DictVectorizer
from . import FeatureVectorizer
from . import Identity
from . import OneHotEncoder
from . import Regressor
from . import TensorToLabel
from . import TensorToProbabilityMap
