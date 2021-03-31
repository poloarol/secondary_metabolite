""" random_sequence.py """

import os
import random
import textwrap
from dataclasses import dataclass, field
from typing import List

import biovec as bv
import numpy as np
import pandas as pd

random.seed(1042)

@dataclass 
class RandomSequence(object):
    """ """

    average_length: int
    number: int
    extra: int
    ALPHABET: List = field(default_factory=list)
    length: int = 0


    def __post_init__(self):
        """ """

        self.ALPHABET = ['A', 'A', 'A', 'A', 'A', 'A', 'A',
                         'R', 'R', 'R', 'R',
                         'N', 'N', 'N', 'N',
                         'D', 'D', 'D', 'D', 'D', 'D',
                         'C', 'C', 'C',
                         'G', 'G', 'G', 'G', 'G', 'G', 'G',
                         'H', 'H', 'H',
                         'I', 'I', 'I', 'I',
                         'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                         'K', 'K', 'K', 'K', 'K', 'K', 'K',
                         'M', 'M',
                         'F', 'F', 'F', 'F',
                         'P', 'P', 'P', 'P', 'P',
                         'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S',
                         'T', 'T', 'T', 'T', 'T', 'T',
                         'W',
                         'Y', 'Y', 'Y',
                         'V', 'V', 'V', 'V', 'V', 'V', 'V'
                        ]
    

    def generatesequence(self):
        bgcs: str = ''
        bgcs_vector: List = []
        model = bv.models.load_protvec(os.path.join(os.getcwd(), 'bgc/embedding/uniprot2vec.model'))
        for i in range(self.number):
            self.length = self.average_length + random.randint(0, self.extra)
            bgc: str = ''
            for i in range(self.length):
                aa: str = random.choice(self.ALPHABET)
                bgc = aa + bgc
            name: str = '>rnd_{}'.format(self.length)
            bgc_vector = np.array(model.to_vecs(bgc))
            bgc_vector = bgc_vector.ravel()
            bgcs_vector.append(bgc_vector)
            bgc = '\n'.join(textwrap.wrap(bgc, 70))
            if bgcs:
                bgcs = '{}\n{}\n{}'.format(name,bgc,bgcs)
            else:
                bgcs = '{}\n{}'.format(name, bgc)
        data = pd.DataFrame(bgcs_vector, columns=[x for x in range(300)])
        data.to_csv(os.path.join(os.getcwd(), 'rnd_{0}.csv'.format(self.average_length)), index=False, header=False,sep=',')
        # with open(os.path.join(os.getcwd(), 'seqs-6500.fasta'), 'w') as tmp:
        #     tmp.write(bgcs)
        # return data