"""
Created on Sat Feb 28 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of a basic HMM
"""

import numpy as np
from random import random

def weighted_choice(probability_list):
    """
    Simulate a value from a probability
    """
    random_number = random()
    state_ind = -1
    cumprob = 0
    for prob in probability_list:
        cumprob += prob
        state_ind += 1
        if cumprob >= random_number:
            break

    return state_ind

class HiddenMarkovModel:
    """
    Implements a basic Hidden Markov Model with a certain
    number of states and emission symbols
    """

    def __init__(self, states, symbols):
        """
        Initiate a HMM model:
            - states: a list with the states or the number of states
            - symbols: a list of symbols of the number of symbols
        """
        if type(states) == int:
            self._states = range(states)
            self._n_states = states
        elif type(states) == list:
            self._states = states
            self._n_states = len(states)
        else:
            print '"states" have to be a list or the number of states'
            raise AttributeError

        if type(symbols) == int:
            self._symbols = range(symbols)
            self._n_symbols = symbols
        elif type(symbols) == list:
            self._symbols = symbols
            self._n_symbols = len(symbols)
        else:
            print '"symbols" have to be a list or the number of states'
            raise AttributeError
        self._symbol_mapping = {s:i for i, s in enumerate(self._symbols)}

    def set_emission_probabilities(self, emission_probs):
        if emission_probs.shape[0] == self._n_symbols and \
            emission_probs.shape[1] == self._n_states:
            self._emission_probs = emission_probs
            self._log_emission_probs = np.log(emission_probs)
        else:
            print 'Provided matrix is of wrong shape!'
            raise IndexError

    def set_transition_probabilities(self, transition_probs):
        if transition_probs.shape[0] == self._n_states and \
            transition_probs.shape[1] == self._n_states:
            self._transition_probs = transition_probs
            self._log_transition_probs = np.log(transition_probs)
        else:
            print 'Provided matrix is of wrong shape!'
            raise IndexError

    def set_starting_probabilities(self, starting_probs):
        if len(starting_probs) == self._n_states:
            self._starting_probs = starting_probs
            self._log_starting_probs = np.log(starting_probs)
        else:
            print 'Provided matrix is of wrong shape!'
            raise IndexError

    def get_emission_probabilities(self):
        return self._emission_probs

    def get_transition_probabilities(self):
        return self._transition_probs

    def get_starting_probabilities(self):
        return self._starting_probs

    def simulate_chain(self, length):
        """
        Simulates a hidden markov model,
        returns hidden and observed variabels as two lists
        """
        state_nr = weighted_choice(self._starting_probs)
        hidden_chain = [self._states[state_nr]]
        symbols_chain = [self._symbols[weighted_choice(\
                self._emission_probs[:, state_nr])]]
        while len(hidden_chain) < length:
            state_nr = weighted_choice(self._transition_probs[state_nr,:])
            hidden_chain.append(self._states[state_nr])
            symbols_chain.append(self._symbols[weighted_choice(\
                    self._emission_probs[:, state_nr])])
        return hidden_chain, symbols_chain

    def generate_max_product(self, sequence, keep_pointer=True):
        """
        Fills in the dynamic programming matrix for viberti
        """
        symbol_indices = map(lambda seq:self._symbol_mapping[seq],list(sequence))
        # fill the matrix
        dynamic_prog_matrix = np.zeros((self._n_states, len(sequence)))
        for state_ind, state in enumerate(self._states):
            init_symbol_ind = symbol_indices[0]
            dynamic_prog_matrix[state_ind, 0] = \
                    + self._log_starting_probs[state_ind]\
                    + self._log_emission_probs[init_symbol_ind, state_ind]
        if keep_pointer:
            pointers = np.zeros((self._n_states, len(sequence)-1), dtype=int)
        for position, symb_ind in enumerate(symbol_indices[1:], 1):
            for state_ind, state in enumerate(self._states):
                gains = [dynamic_prog_matrix[s_prev, position - 1]\
                        + self._log_transition_probs[s_prev, state_ind]\
                        + self._log_emission_probs[symb_ind, state_ind]\
                        for s_prev in range(self._n_states)]
                dynamic_prog_matrix[state_ind, position] = np.max(gains)
                if keep_pointer:
                    pointers[state_ind, position-1] = np.argmax(gains)
        if keep_pointer:
            return dynamic_prog_matrix, pointers
        else:
            return dynamic_prog_matrix

    def viterbi(self, sequence, return_log_prob=False):
        '''
        Calculate the most likely hidden states for a given sequence
        set return_log_prob to true to also get the log conditional probabilty
        '''
        dynamic_prog_matrix, pointers = self.generate_max_product(sequence, True)
        state_ind = np.argmax(dynamic_prog_matrix[:, -1])
        inferred_hidden_states = [self._states[state_ind]]
        pos = -1
        while len(inferred_hidden_states) < len(sequence):
            state_ind = pointers[state_ind, pos]
            state = self._states[state_ind]
            inferred_hidden_states = [state] + inferred_hidden_states
            pos -= 1
        if not return_log_prob:
            return inferred_hidden_states
        else:
            return inferred_hidden_states, np.max(dynamic_prog_matrix[:,-1])

    def generate_sum_product(self, sequence):
        '''
        Calculate dynamic programming matrix for forward algorithms
        '''
        symbol_indices = map(lambda seq:self._symbol_mapping[seq], list(sequence))
        # fill the matrix
        dynamic_prog_matrix = np.zeros((self._n_states, len(sequence)))
        for state_ind, state in enumerate(self._states):
            init_symbol_ind = symbol_indices[0]
            dynamic_prog_matrix[state_ind, 0] = \
                    self._starting_probs[state_ind]\
                    * self._emission_probs[init_symbol_ind, state_ind]
        for position, symb_ind in enumerate(symbol_indices[1:], 1):
            for state_ind, state in enumerate(self._states):
                gains = [dynamic_prog_matrix[s_prev, position - 1]\
                        * self._transition_probs[s_prev, state_ind]\
                        * self._emission_probs[symb_ind, state_ind]\
                        for s_prev in range(self._n_states)]
                dynamic_prog_matrix[state_ind, position] = np.sum(gains)
        return dynamic_prog_matrix

    def forward_algorithm(self, sequence):
        '''
        Calculate the probability of the sequence according to the HMM
        '''
        dynamic_prog_matrix = self.generate_sum_product(sequence)
        return dynamic_prog_matrix[:, -1].sum()

if __name__ == '__main__':
    symbols = ['H', 'T']
    states = ['F', 'B']

    emis_probs = np.array([[0.5, 0.1], [0.5, 0.90]])
    trans_probs = np.array([[0.8, 0.2], [0.2, 0.8]])

    HMM = HiddenMarkovModel(states, symbols)
    HMM.set_emission_probabilities(emis_probs)
    HMM.set_transition_probabilities(trans_probs)
    HMM.set_starting_probabilities(np.array([1, 0]))
    hidden, symbols = HMM.simulate_chain(100)
    print ''.join(symbols)
    print ''.join(hidden)

    inferred, log_cond_prob = HMM.viterbi(symbols, return_log_prob=True)
    print ''.join(inferred), '\n'

    probability = HMM.forward_algorithm(symbols)

    print 'log likelihood most likely sequence is %.3f, total log likelihood is %.3f'\
            %(log_cond_prob, np.log(probability))
