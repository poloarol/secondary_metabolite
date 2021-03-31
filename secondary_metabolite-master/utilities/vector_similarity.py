""" """

import os

import numpy as np
import pandas as pd

from typing import Any, List

from dataclasses import dataclass
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class VectorSimilarity(object):

    vector: Any


    def cosine_similarity(self, vector_1: List):

        numerator = np.dot(self.vector, vector_1)
        denominator = np.linalg.norm(self.vector) * np.linalg.norm(vector_1)

        return numerator/denominator


vectors = pd.read_csv(os.path.join(os.getcwd(), '..\\mibig_average_vectors.csv'), delimiter=',', header=None)

print(cosine_similarity(vectors, vectors))


# for vector in vectors:
#     vector_similarity = VectorSimilarity(vector[1:])

#     for v in vectors:

#         cos_sim = vector_similarity.cosine_similarity(v[1:])

#         print('{} - {} - {}'.format(vector[0], v[0], cos_sim))

