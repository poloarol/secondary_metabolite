from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

from Bio import SeqIO

import os


class CustomBlast(object):
    
    def customblast(self, accession: str):
        handle = NCBIWWW.qblast("blastp", "protein", accession)
        results = NCBIXML.parse(handle)

        print(results)
    

    def write(self, accesion, alignment):
        with open(os.path.join(os.getcwd(), '../tmp/tmp/{}.fasta'.format(accesion)), 'w') as handle:
            SeqIO.write(alignment, handle, 'fasta')


custom = CustomBlast()
custom.customblast('AMW76289.1')