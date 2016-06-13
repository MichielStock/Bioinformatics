# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 2016

@author: michielstock

Module for generating samples from alignments of sequences (proteins or DNA/RNA)
"""

from Bio import AlignIO
from collections import Counter
import pandas as pd

class PositionSpecificSubstitution:
    """
    A simple class for sampling sequences from an alignment file
    """
    def __init__(self, alignment, use_gaps=True, pseudocounts=1):
        self._pseudocounts = pseudocounts
        self._use_gaps = use_gaps
        AA_freq_pos = []
        self._alignment_length = alignment.get_alignment_length()
        for i in range(self._alignment_length):
            AA_freq_pos.append(Counter(alignment[1:,i]))
        # save as pandas dataframe
        self.pssm = pd.DataFrame(AA_freq_pos).T
        # remove missing values (char not encountered)
        self.pssm = self.pssm.fillna(0)
        self.pssm += pseudocounts  # add pseudocounts
        if not use_gaps:  # remove the '-' gap character
            self.pssm = self.pssm[1:]
        self.pssm /= self.pssm.sum()  # normalize
        self.pssm_cumulative = self.pssm.cumsum(0)  #



alignment = AlignIO.read("muscle_cons_refs.txt", "clustal")
alignment.sort()

print alignment

AA_freq_pos = []
n = alignment.get_alignment_length()

for i in range(n):
    if alignment[0,i] is not '-':
        AA_freq_pos.append(Counter(alignment[1:,i]))


AA_table = pd.DataFrame(AA_freq_pos).T
AA_table = AA_table.fillna(0)  # remove NA
AA_table = AA_table[1:]  # remove empty char

AA_table /= AA_table.sum()

print AA_table.sum()

print AA_table

AA_table.var().plot()

AA_table.to_csv('AA_distibution_consensus.csv')

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(AA_table.values)
plt.savefig('pos_AA.pdf')
