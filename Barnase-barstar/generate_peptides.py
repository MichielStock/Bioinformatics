"""
Created on Wednesday 29 November 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Parses the Skempi (https://life.bsc.es/pid/mutation_database/database.html)
database and returns a FASTA file with all the mutated sequences of barnase and
their activity towards barstar.
"""

from Bio import SeqIO
from Bio.Alphabet import ProteinAlphabet
from math import log10
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from sys import path
path.append('../Protein_structures')
from structures import StructurePDB

def parse_mutation(mutation):
    wt_AA, chain, position, mut_AA = mutation[0], mutation[1],\
                                            int(mutation[2:-1]), mutation[-1]
    return wt_AA, chain, position, mut_AA

def get_mutated_seq(chains, mutations):
    pass

# load mutations
skempi = pd.read_csv('skempi.csv', sep=';')

mutations_with_activity = [(mut.split(','), log10(dh_mut/dh_wt)) for (mut, dh_mut, dh_wt) in\
                                    skempi[['Mutation(s)_cleaned',
                                    'Affinity_mut (M)', 'Affinity_wt (M)']].values]

barnstar_chains = {}

with open('1brs.pdb', 'r') as fh:
    for record in SeqIO.parse(fh, format='pdb-seqres'):
        barnstar_chains[record.annotations['chain']] = record.seq

if __name__ == '__main__':
    # check the proteins
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser()
    pdb_bb = parser.get_structure("barnstar-barnase", "1brs.pdb")

    struct_bb = StructurePDB(pdb_bb)
    fig, dist = struct_bb.make_contact_map()

    fig.savefig('Figures/contact_map.png')

    # plot mutations
    fig, (axA, axD) = plt.subplots(nrows=2)

    for mutations, act in mutations_with_activity:
        for mut in mutations:
            wt_AA, chain, position, mut_AA = parse_mutation(mut)
            ax = axA if chain=='A' else axD
            ax.text(x=position, y=act, s=mut_AA)
            ax.scatter(position, act, alpha=0.2)
    fig.savefig('Figures/mutations.png')
