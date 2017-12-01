# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:24:25 2016

@author: michielstock
"""

from heapq import heappop, heappush, heapify


def naive_suffix_array_creation(string):
    """
    Creates a suffix array by using python's sort function
    Very inefficient in time and memory
    """
    indexed_suffixes = [(string[i:], i) for i in range(len(string))]
    indexed_suffixes.sort()
    return zip(*indexed_suffixes)[1]


def multiple_suffix_array(suffix_arrays, strings):
    n_strings = len(strings)
    positions = [0] * n_strings
    multiple_suffix_array = []
    sequence_id = []
    suffixes_to_add = [(strings[i][suffix_arrays[i][0]:], i,
                        suffix_arrays[i][0]) for i in range(n_strings)]
    heapify(suffixes_to_add)
    while len(suffixes_to_add):
        _, seq_ind, suffix_ind = heappop(suffixes_to_add)
        multiple_suffix_array.append(suffix_ind)
        sequence_id.append(seq_ind)
        if positions[seq_ind] < len(strings[seq_ind]) - 1:
            positions[seq_ind] += 1
            pos = positions[seq_ind]
            heappush(suffixes_to_add,
                     (strings[seq_ind][suffix_arrays[seq_ind][pos]:], seq_ind,
                        suffix_arrays[seq_ind][positions[seq_ind]]))
    return multiple_suffix_array, sequence_id






if __name__ == '__main__':
    
    string1 = 'iefubfewiubfiuewbfiuewbiufeiufwbibewfbewfssbfibsdbvbvbiubvsduibu'
    string2 = 'ubwvwbibvubisuvinosd'
    string3 = 'vytscubaimonuibyvtcryvubiyvtcrxetcyvubino'
    
    # single SA    
    suffix_array = naive_suffix_array_creation(string)
    
    # multiple SA
    strings = [string1, string2, string3]
    suffix_arrays = [naive_suffix_array_creation(string) for string in strings]
    print multiple_suffix_array(suffix_arrays, strings)