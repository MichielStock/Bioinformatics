"""
Created on Tue Jul 21 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Most frequent k characters similarity

Compares two strings based on their most frequent used characters/tuples

https://en.wikipedia.org/wiki/Most_frequent_k_characters
"""

from collections import Counter

def most_frequent_k_hashing(input_string, K, substring_length=1):
    """
    Hashes a string into its K most occuring substrings of a certain length
        (default is 1)
    """
    if substring_length == 1:
        hashing = Counter(input_string).most_common(K)
    else:
        len_str = len(input_string)
        input_string_in_substring = [input_string[i:i+substring_length] for\
                i in range(len_str - substring_length)]
        hashing = Counter(input_string_in_substring).most_common(K)
    hasing_dict = {k:v for k,v in hashing}
    return hasing_dict

def most_frequent_k_similarity(input_hash1, input_hash2):
    """
    Calculates the most frequent k similarity based on two hashes
    by taking the sum of the minimal occurences of the strings
    """
    similarity = 0
    for k1, v1 in input_hash1.iteritems():
        if input_hash2.has_key(k1):
            similarity += min(v1, input_hash2[k1])
    return similarity

if __name__ == '__main__':
    string1 = 'LCLYTHIGRNIYYGSYLYSETWNTGIMLLLITMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLV'
    string2 = 'EWIWGGFSVDKATLNRFFAFHFILPFTMVALAGVHLTFLHETGSNNPLGLTSDSDKIPFHPYYTIKDFLG'

    print most_frequent_k_hashing(string1, 2, 1)
    print most_frequent_k_hashing(string1, 2, 2)

    for len_ss in [1, 2, 3]:
        h1 = most_frequent_k_hashing(string1, 2, len_ss)
        h2 = most_frequent_k_hashing(string2, 2, len_ss)

        sim = most_frequent_k_similarity(h1, h2)

        print 'K=2 similarity for substr. length of %s is %s'%(len_ss, sim)
