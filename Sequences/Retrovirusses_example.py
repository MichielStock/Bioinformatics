"""
Created on Tue Jul 21 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Example of virus similarity on retrovirusses
"""

from Bio import Entrez, SeqIO
from MostFrequentK import most_frequent_k_hashing, most_frequent_k_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# enter mail for Entrez
Entrez.email = 'emailadress@mail.com'


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

n_genomes = len(genomes)

# save names

fh = open('Retro_virusses_names.txt', 'w')
for name in names:
    fh.write(name + '\n')
fh.close()

# make heatmaps of the similarities

fig, axes = plt.subplots(nrows=2, ncols=3)

Ks = [5, 10, 20]
substr_lengths = [3, 5]

for i in range(3):
    K = Ks[i]
    for j in range(2):
        substr_len = substr_lengths[j]
        ax = axes[j][i]

        # hash the genomes
        hashes = [most_frequent_k_hashing(genome, K, substr_len)\
                        for genome in genomes]

        similarity_matrix = np.zeros((n_genomes,n_genomes), dtype=int)

        for pos1 in range(n_genomes):
            for pos2 in range(pos1 + 1):
                similarity = most_frequent_k_similarity(hashes[pos1],
                                                hashes[pos2])
                similarity_matrix[pos1, pos2] = similarity
                similarity_matrix[pos2, pos1] = similarity

        ax.imshow(similarity_matrix, cmap='hot')
        ax.set_title('K=%s feq. sim.\n(substr. len.=%s)'%(K, substr_len))

fig.show()

# on K = 5, substr_len = 3

hashes = [most_frequent_k_hashing(genome, 5, 3) for genome in genomes]
similarity_matrix = np.zeros((n_genomes,n_genomes), dtype=int)
for pos1 in range(n_genomes):
    for pos2 in range(pos1 + 1):
        similarity = most_frequent_k_similarity(hashes[pos1],
                                                hashes[pos2])
        similarity_matrix[pos1, pos2] = similarity
        similarity_matrix[pos2, pos1] = similarity

virus_sim = pd.DataFrame(similarity_matrix)
virus_sim.index = names

virus_sim.to_csv('Virus_similarity.csv')
