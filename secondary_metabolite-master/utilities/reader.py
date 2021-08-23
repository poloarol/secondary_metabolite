"""reader.py -- provides methods to read and write both GenBank and Fasta files"""

import os
import textwrap

from dataclasses import dataclass
from typing import Any, Dict

import biovec as bv
import numpy as np

from Bio import SeqIO

@dataclass
class ReadGB(object):
    """

    Read GenBank files, and allow for conversion into numerical vectors for learning.

    Parameters
    ----------

    file: Gb file
    model: biovec model (NLP)
    seq: vectors of the Gene Cluster

    """

    file: Any
    model =  bv.models.load_protvec(os.path.join(os.getcwd(), 'bgc\\models\\biovec\\uniprot2vec.model'))
    bgc_seq: str = ''
    seq: str = np.array([])

    def __post_init__(self):
        """ Reads the GB file and performs vectorial conversion. """

        self.records = SeqIO.parse(open(self.file, encoding='utf-8', errors='ignore'), 'genbank')
        try:
            for record in self.records:
                for feature in record.features:
                    if feature.type == 'CDS':
                        tmp: str = feature.qualifiers['translation'][0].strip()
                        self.bgc_seq = ''.join([self.bgc_seq, tmp])
                        if self.seq.size == 0:
                            self.seq = np.array(self.model.to_vecs(tmp))
                        else:
                            bgc = np.array(self.model.to_vecs(tmp))
                            self.seq = self.seq + bgc
                        
        except Exception as e:
            print(e.__repr__())

    def to_fasta(self, fasta_path: str, identifier: str):

        """
        Write GB file into fastas format.
        """
        with open(fasta_path, 'w') as output_handle:
            output_handle.write('>{}\n{}'.format(identifier, textwrap.fill(self.bgc_seq, width=60)))
    
    def get_vector(self):
        return self.seq

