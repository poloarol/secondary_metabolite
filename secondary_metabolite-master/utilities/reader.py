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
    record: Any = None
    bgc: Dict = np.array([])
    model =  bv.models.load_protvec(os.path.join(os.getcwd(), 'bgc\\models\\biovec\\uniprot2vec.model'))
    store: str = ''
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
                        
                        if 'X' in tmp:
                            aas = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 
                                    'R', 'R', 'R', 'R', 'N', 'N', 
                                    'D', 'D', 'C', 'C', 'E', 'E',
                                    'Q', 'Q', 'Q', 'Q', 'Q', 'W',
                                    'G', 'G', 'G', 'G', 'G', 'G', 'G',
                                    'H', 'H', 'H', 'I', 'I', 'I', 'I',
                                    'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
                                    'K', 'K', 'K', 'K', 'K', 'K', 'K',
                                    'M', 'M', 'F', 'F', 'F', 'F',
                                    'P', 'P', 'P', 'P', 'P',
                                    'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S',
                                    'T', 'T', 'T', 'T', 'Y', 'Y', 'Y',
                                    'V', 'V', 'V', 'V', 'V', 'V', 'V']
                            aa = random.choice(aas)
                            tmp  = tmp.replace('X', aa)
                        if 'B' in tmp:
                            aas = ['D', 'N']
                            aa = random.choice(aas)
                            tmp  = tmp.replace('B', aa)
                        if 'Z' in tmp:
                            aas = ['E', 'Q']
                            aa = random.choice(aas)
                            tmp  = tmp.replace('Z', aa)
                        if 'J' in tmp:
                            aas = ['I', 'L']
                            aa = random.choice(aas)
                            tmp  = tmp.replace('J', aa)
                        
                        if len(tmp) < 4:
                            self.store = tmp
                        else:
                            if self.store:
                                seq = ''.join([self.store, tmp])
                                self.calc_seq_by_seq(seq)
                                self.store = ''
                            else:
                                self.calc_seq_by_seq(tmp)
                        
        except Exception as e:
            print(e.__repr__())

    def to_fasta(self, fasta_path: str):

        """
        """
        with open(fasta_path, 'w') as output_handle:
            output_handle.write(self.seq)
    
    def get_vector(self):

        return self.bgc
    
    def calc_seq_by_seq(self, seq: str):
        """
        """
        # print(np.array(self.model.to_vecs(seq).shape))
        tmp = np.array(self.model.to_vecs(seq), dtype='object')
        if self.bgc.size != 0:
            self.bgc = np.add(self.bgc, tmp, dtype='object')
        else:
            self.bgc = tmp

