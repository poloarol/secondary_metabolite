"""reader.py -- provides methods to read and write both GenBank and Fasta files"""

import os
import random

from dataclasses import dataclass
from typing import Any, Dict

import biovec as bv
import numpy as np

from Bio import SeqIO

@dataclass
class ReadGB(object):
    """
    """

    file: Any
    cluster: str = ''
    model =  bv.models.load_protvec(os.path.join(os.getcwd(), 'bgc\\models\\biovec\\uniprot2vec.model'))
    seq: str = ''

    def __post_init__(self):
        """
        """

        self.records = SeqIO.parse(open(self.file, encoding='utf-8', errors='ignore'), 'genbank')
        try:
            for record in self.records:
                for feature in record.features:
                    if feature.type == 'CDS':
                        tmp: str = feature.qualifiers['translation'][0].strip()
                        self.seq = f'{self.seq}{tmp}'
                        
        except Exception as e:
            print(e.__repr__())

    def to_fasta(self, fasta_path: str):

        """
        """
        with open(fasta_path, 'w') as output_handle:
            output_handle.write(self.seq)
    
    def get_vector(self):
        
        try:
            bgc_vector = np.array(self.model.to_vecs(self.seq))
            return bgc_vector
        except Exception as e:
            print(e.__repr__)
            
            return np.array([])

