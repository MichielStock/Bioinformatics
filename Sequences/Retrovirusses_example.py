"""
Created on Tue Jul 21 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Example of virus similarity on retrovirusses
"""

from Bio import Entrez, SeqIO
from Mos

Entrez.email = 'michielfmstock@gmail.com'


# Read the accession numbers

fh = open('retrovirusses_accession', 'r')
acc_nrs = []
for l in fh:
    acc_nrs.append(l.rstrip())
fh.close()

names = []  # name of the genome
genomes = []  # genomes

for acc in acc_nrs:
    handle = Entrez.efetch(db="nucleotide", rettype="fasta", retmode="text", id=acc)
    seq_record = SeqIO.read(handle, "fasta")
    names.append(seq_record.description)
    genomes.append(str(seq_record.seq))
