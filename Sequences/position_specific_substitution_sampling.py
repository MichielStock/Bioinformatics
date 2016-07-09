# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 2016

@author: michielstock

Module for generating samples from alignments of sequences (proteins or DNA/RNA)

Includes functions for optimizing the sequences according to a given criterium
"""

from Bio import AlignIO
from collections import Counter
import pandas as pd
from random import random
from math import log10, ceil
import numpy as np

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

    def sample_sequence(self, keep_gaps=False):
        """
        Samples a sequence based on the position specific substitution matrix
        """
        sampled_sequence = ''
        for i in range(self._alignment_length):
            character = sample_discrete(self.pssm_cumulative[i],
                            self.pssm.index)
            if keep_gaps or character != '-':
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

    def set_internal_sequence(self, sequence=None):
        """
        Set an internal sequence, make a random one if None is given
        """
        if sequence is None:
            self._internal_sequence = self.sample_sequence(keep_gaps=True)
        else:
            self._internal_sequence = sequence

    def get_internal_sequence(self):
        """
        Gets the internal random sequence
        """
        return self._internal_sequence.replace('-', '')

    def mutate_internal_sequence(self, pmut=0.01, return_sequence=False):
        """
        Mutates the internal sequence, either change it or return it
        """
        mutated_sequence = ''
        for i in range(self._alignment_length):
            if random() > pmut:
                character = self._internal_sequence[i]
            else:
                character = sample_discrete(self.pssm_cumulative[i],
                            self.pssm.index)
            mutated_sequence += character
        if return_sequence:
            return mutated_sequence
        else:
            self._internal_sequence = mutated_sequence

    def simulated_annealing(self, Tmax, Tmin, pmut, r, kT,
                                        scoring_function, keep_samples=False):
        """
        Uses simulated annealing to optimize the internal sequence to
        MAXIMIZE a scoring function

            Inputs:
                - Tmax : maximum (starting) temperature
                - Tmin : minimum (stopping) temperature
                - pmut : probability of mutating an amino acid in the sequence
                - r : rate of cooling
                - kT : number of iteration with fixed temperature
                - scoring_function : the scoring function used for the sequence

            Outputs:
                - sequence : best found sequence
                - fbest : best scores obtained through the iterations
                - samples : samples of the sequences
        """
        temp = Tmax
        fstar = scoring_function(self.get_internal_sequence())
        fbest = [fstar]
        samples = []

        while temp > Tmin:
            for _ in range(kT):
                new_sequence = self.mutate_internal_sequence(pmut, True)
                fnew = scoring_function(new_sequence.replace('-', ''))
                if np.exp(-(fstar - fnew) / temp) > random():
                    self.set_internal_sequence(new_sequence)
                    fstar = fnew
            fbest.append(fstar)
            temp *= r
            if keep_samples:
                samples.append(self.get_internal_sequence())
        return self.get_internal_sequence(), fbest, samples

if __name__ == '__main__':

    # read an alignment
    alignment = AlignIO.read("muscle_cons_refs.txt", "clustal")

    # sample a protein from the alignment
    pss_sampler = PositionSpecificSubstitutionSampler(alignment)
    print(pss_sampler.sample_sequence())

    pss_sampler.set_internal_sequence()

    # optimize the sequences for A/H count
    count_A_H = lambda seq : seq.count('A') + seq.count('H')

    sequence, fbest, samples = pss_sampler.simulated_annealing(100, 0.01,
                0.01, 0.9, 10, count_A_H, True)

    for step, seq in enumerate(samples):
        print('Step {}: A={}, H={}'.format(step + 1, seq.count('A'),
                                seq.count('H')))
