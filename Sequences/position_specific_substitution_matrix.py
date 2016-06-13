# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 2016

@author: michielstock

Module for generating samples from alignments of sequences (proteins or DNA/RNA)
"""

from Bio import AlignIO
from collections import Counter
import pandas as pd
from random import random
import matplotlib.pyplot as plt

def sample_discrete(cumulative, values):
    """
    Samples from a discrete distribution, given the cumulative distribution
    and the corresponding values
    """
    assert len(cumulative) == len(values)
    random_number = random()
    for cum_prob, value in zip(cumulative, values):
        if cum_prob >= random_number:
            return value

class PositionSpecificSubstitutionSampler:
    """
    A simple class for sampling sequences from an alignment file
    """
    def __init__(self, alignment, use_gaps=True, pseudocounts=1):
        alignment.sort()
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
        self.pssm_cumulative = self.pssm.cumsum(0)  # cumulative for sampling

    def sample_sequence(self):
        """
        Samples a sequence based on the position specific substitution matrix
        """
        sampled_sequence = ''
        for i in range(self._alignment_length):
            character = sample_discrete(self.pssm_cumulative[i],
                            self.pssm.index)
            if character != '-':
                sampled_sequence += character
        return sampled_sequence

    def get_pssm(self, cumulative=False):
        """
        Returns the position specific substitution matrix
        """
        if cumulative:
            return self.pssm_cumulative
        else:
            return self.pssm


if __name__ == '__main__':
    alignment = AlignIO.read("muscle_cons_refs.txt", "clustal")

    pss_sampler = PositionSpecificSubstitutionSampler(alignment)
    print(pss_sampler.sample_sequence())
