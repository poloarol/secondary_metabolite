""""""

import os
import time
import subprocess

from typing import List
from typing import Tuple
from typing import NamedTuple
from dataclasses import field
from dataclasses import dataclass
from collections import namedtuple

import prody

from Bio import Entrez 
from Bio import SeqIO

Entrez.email = 'adjon081@uottawa.ca'

@dataclass
class Protein(object):
    """
    Stores information relating to a protein as presented in a GB file

    Paramters
    ---------

    protein_id (str): File identifier within the Protein DB on NCBI
    product (str): Name of protein
    translation (str): AA sequence
    nucleotide (str): Nucleotide sequence
    location (Tuple): stores location of nucleotide sequence within the genome
    strand (int): Direction of protein i.e. whether it is on the coding or non-coding strand
    """

    protein_id: str
    product: str
    translation: str
    nucleotide: str
    location: Tuple
    strand: int


@dataclass
class Operon(object):
    """
    Provides a structure to keep clusters of protein together i.e.
    All genes products (proteins) which are under the influences 
    of single promoter.

    Parameter
    ---------

    count (int): identifies the number of genes which are
                    primary metabolites within the operon
    direction (int): direction of all proteins within the operon
    OPERON (List): List which stores all proteins within the operon
    """

    count: int = 0
    direction: int = 0
    OPERON: List = field(default_factory=list)

    def add(self, protein):
        """ Adds a new protein to the operon """
        self.OPERON.append(protein)

    def increment(self):
        """ Adds when a new protein relating to a primary metabolite is added """
        self.count = self.count + 1
    
    def getcount(self):
        return self.count


@dataclass
class Genome(object):
    """
    Stores information relating to the genome of a given organism.
    Proteins are grouped into operons, which are further stored into
    a list.

    Parameters
    ----------

    accession (str): Organism accession number as identified by Nucleotide
                        DB or GenBank
    """

    accession: str
    db: str = os.path.join(os.getcwd(), 'tmp/minimal/minimalgenome.hmm')
    GENOME: List = field(default_factory=list)

    def __post_init__(self):
        gb = Entrez.efetch(db='nucleotide', id=self.accession, rettype='gb', retmode='text')
        records = SeqIO.read(gb, 'genbank')
        operon = Operon()

        for i, feature in enumerate(records.features):

            strand: int
            product: str
            protein_id: str
            location: Tuple
            nucleotide: str
            notfound: bool
            translation: str

            if 'CDS' == feature.type:
                try:
                    protein_id = feature.qualifiers['protein_id'][0]
                except KeyError:
                    pass
                try:
                    product = feature.qualifiers['product'][0]
                except KeyError:
                    pass
                try:
                    translation = feature.qualifiers['translation'][0]
                except KeyError:
                    pass
                try:
                    location = (feature.location.nofuzzy_start, feature.location.nofuzzy_start)
                except KeyError:
                    pass
                nucleotide = records.seq[location[0] : location[1]]
                strand = int(feature.strand)
                protein = Protein(protein_id, product, translation, nucleotide, location, strand)

                if protein_id:
                    operon = self.add_operon(strand, operon, protein)
                    notfound = self.query(protein_id)

                if notfound:
                    operon.increment()
        
    def add_operon(self, strand, operon, protein):
        """ Adds proteins to the operon """

        if self.GENOME:
            if operon.direction == strand:
                operon.add(protein)
            else:
                self.add(operon)
                operon = Operon(direction=strand)
                operon.add(protein)
        else:
            operon.direction = strand
            operon.add(protein)
            self.add(operon)
    
        return operon
        
    def add(self, operon):
        self.GENOME.append(operon)
        
    def query(self, protein_id: str):
        """ Searches the HMM DB to determine whether protein sequence makes a hit """

        protein: str = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
        record = SeqIO.read(protein, 'fasta')

        queries: str = os.path.join(os.getcwd(), 'tmp/tmp/hmmqueries/{}.txt'.format(protein_id))
        path: str = os.path.join(os.getcwd(), 'tmp/tmp/results/{}.txt'.format(protein_id))

        with open(queries, 'w') as handle:
            SeqIO.write(record, handle, 'fasta')
                
        subprocess.Popen(['hmmscan', '-E', '1.001', '--cpu', '5', '-o', path, self.db, queries], stdout=subprocess.PIPE)

        time.sleep(3)

        with open(path, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                if '[No targets detected that satisfy reporting thresholds]' in line:
                    return False
        return True
    
    def check(self):
        return self.__next__() and self.__prev__()
    

    def __next__(self, counter: int):
        val: bool

        if len(self.GENOME) == counter:
            pass
        elif counter == 0:
            pass
        else:
            pass

        return val

    def __prev__(self, counter: int):
        val: bool

        if len(self.GENOME) == counter:
            pass
        elif counter == 0:
            pass
        else:
            pass

        return val

        return val
    

    def show(self):
        with open(os.path.join(os.getcwd(), 'tmp/tmp/streptomyces/{}.txt'.format(self.accession)), 'w') as f:
            for op in self.GENOME:
                if op.count < 2:
                    f.write('*************************************************\n')
                    for p in op.OPERON:
                        f.write('{0}, {1}, {2}\n'.format(p.protein_id, p.product, p.strand))
                    f.write('*************************************************\n')

@dataclass
class MinimalGenome():
    """
    Generates a Minimal Genome from {{ CP01490.1 }}
    """
    accession: str = 'CP01490.1'

    def __post_init__(self):

        gb = Entrez.efetch(db='nucleotide', id=self.accession, rettype='gbwithparts', retmode='text')
        records = SeqIO.read(gb, 'genbank')

        for i, feature in enumerate(records.features):
            if 'CDS' == feature.type:
                protein_id = feature.qualifiers['protein_id'][0]
                self.save(protein_id)

    
    def save(self, protein_id:str):
        """ Builds the protein sequence's HMM and stores it to file """
        protein: str = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
        record = SeqIO.read(protein, 'fasta')
        pfam: str
        with open(os.path.join(os.getcwd(), 'notfound.txt'), 'a') as f:
            try:
                pfam = prody.searchPfam(str(record.seq), timeout=300)
                time.sleep(10)
                for key in pfam:
                    pfam_ids = pfam[key]['accession'].split(' ')
                    for pfam_id in pfam_ids:
                        time.sleep(10)
                        msa = prody.fetchPfamMSA(pfam_id, timeout=300)
                        path: str = os.path.join(os.getcwd(), 'tmp/minimal_genome/{}'.format(pfam_id))
                        subprocess.Popen(['hmmbuild', path, msa], stdout=subprocess.PIPE)
            except Exception as err:
                f.write(pfam if pfam else protein_id)


@dataclass
class SecondaryMetabolites(object):
    """
    """

    def query(self, accession: str):
        pass


                