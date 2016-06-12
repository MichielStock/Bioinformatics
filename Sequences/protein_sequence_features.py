# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:13:18 2016
Last update on Sun Jun 12 2016

@author: michielstock

Create alignment independent features for proteins
"""

import numpy as np
import numba

# amino acid descriptors

Z_1 = {'A': 0.07, 'V' : -2.69, 'L' : -4.19, 'I' : -4.44, 'P' : -1.22,
       'F' : -4.92, 'W': -4.75, 'M': -2.49, 'K': 2.84, 'R' : 2.88, 'H': 2.41,
       'G' :2.23, 'S': 1.96, 'T' : 0.92, 'C' : 0.71, 'Y' : -1.39, 'N' : 3.22,
       'Q' : 2.18, 'D' : 3.64, 'E' : 3.08}

Z_2 = {'A': -1.73, 'V' : -2.53, 'L' : -1.03, 'I' : -1.68, 'P' : 0.88,
       'F' : 1.3, 'W': 3.65, 'M': -.27, 'K': 1.41, 'R' : 2.52, 'H': 1.74,
       'G' :-5.36, 'S': -1.63, 'T' : -2.09, 'C' : 0.71, 'Y' : 2.32,
       'N' : 0.01, 'Q' : 0.53, 'D' : 1.13, 'E' : 0.39}

Z_3 = {'A': 0.09, 'V' : -1.29, 'L' : -0.98, 'I' : -1.03, 'P' : 2.23,
       'F' : 0.45, 'W': 0.85, 'M':-.41, 'K': -3.14, 'R' : -3.44, 'H': 1.11,
       'G' :0.30, 'S': 0.57, 'T' : -1.4, 'C' :4.13 , 'Y' : 0.01, 'N' : 0.84,
       'Q' : -1.14, 'D' : 2.36, 'E' : -0.07}

amino_acids = set(Z_1.keys())

# make a numpy array with the physicochemical properties
# rows are the amino acids, columns are the properties
descriptors = np.zeros((len(Z_1), 3))
amino_acids_inversed_index = {}
for i, AA in enumerate(Z_1.keys()):
    amino_acids_inversed_index[AA] = i
    descriptors[i, :] = [Z_1[AA], Z_2[AA], Z_3[AA]]

def map_seq_to_int(sequence):
    """
    Maps a sequence of amino acids to a numpy array of corresponding integers

    unkown AA are mapped to -1
    """
    sequence_array = np.zeros(len(sequence), dtype=int)
    for i, AA in enumerate(sequence):
        if AA in amino_acids:
            sequence_array[i] = amino_acids_inversed_index[AA]
        else:
            sequence_array[i] = -1
    return sequence_array

@numba.jit
def calc_correlation_feature(sequence_array, lag, prop_ind_1, prop_ind_2,
                                descriptors=descriptors):
    """
    Calculates a single correction feature between between two properties and
    a lag phase
    """
    feature_value = 0
    n = len(sequence_array)
    for i in range( n - lag ):
        if sequence_array[i] != -1 and sequence_array[i + lag] != -1:
            AA_i = sequence_array[i]
            AA_j = sequence_array[i + lag]
            feature_value += ( descriptors[AA_i, prop_ind_1] *\
                        descriptors[AA_j, prop_ind_2] )/( n - lag )
    return feature_value

@numba.jit
def fill_correlation_features(sequence_array, feature_vector, lag_range,
                discriptors=descriptors):
    k = len(lag_range)
    n_AA, p = discriptors.shape
    ind = 0
    for lag in lag_range:
        for prop_i in range(p):
            for prop_j in range(p):
                if prop_i >= prop_j:
                    feature_vector[ind] = calc_correlation_feature(sequence_array,
                    lag, prop_i, prop_j, descriptors)
                    ind += 1
    return feature_vector


def protein_features(sequence, lag_range=range(1, 26),
                    discriptors=descriptors):
    """
    Calculates features of protein sequences based on lagged correlation
    of physicochemical properties of the amino acids
    """
    k = len(lag_range)
    n_AA, p = discriptors.shape
    n_features = (p + (p * (p - 1)) / 2) * k
    feature_vector = np.zeros(n_features)
    sequence_array = map_seq_to_int(sequence)
    feature_vector = fill_correlation_features(sequence_array, feature_vector,
                lag_range, discriptors=descriptors)
    return feature_vector

if __name__ == '__main__':

    # example on
    # RNA-dependent RNA polymerase [Sudan ebolavirus]

    sequence = '''MMATQHTQYPDARLSSPIVLDQCDLVTRACGLYSEYSLNPKLRTCRLPKHIY
    RLKYDAIVLRFISDVPVATIPIDYIAPMLINVLADSKNAPLEPPCLSFLDEIVNYTVQDAAFLNYYMNQI
    KTQEGVITDQLKQNIRRVIHKNRYLSALFFWHDLSILTRRGRMNRGNVRSTWFVTNEVVDILGYGDYIFW
    KIPIALLPMNSANVPHASTDWYQPNIFKEAIQGHTHIISVSTAEVLIMCKDLVTSRFNTLLIAELARLED
    PVSADYPLVDDIQSLYNA
    GDYLLSILGSEGYQIIKYLEPLCLAKIQLCSQYTERKGRFLTQMHLAVIQTLRELLLNRGLKKSQLSKIR
    EFHQLLLRLRSTPQQLCELFSIQKHWGHPVLHSEKAIQKVKNHATVLKALRPIIIFETYCVFKYSVAKHF
    FDSQGTWYSVISDRCLTPGLNSYIRRNQFPPLPMIKDLLWEFYHLDHPPLFSTKIISDLSIFIKDRATAV
    EQTCWDAVFEPNVLGYSPPYRFNTKRVPEQFLEQEDFSIESVLQYAQELRYLLPQNRNFSFSLKEKELNV
    GRTFGKLPYLTRNVQTLCEALLADGLAKAFPSNMMVVTEREQKESLLHQASWHHTSDDFGEHATVRGSSF
    VTDLEKYNLAFRYEFTAPFIKYCNQCYGVRNVFDWMHFLIPQCYMHVSDYYNPPHNVTLENREYPPEGPS
    AYRGHLGGIEGLQQKLWTSISCAQISLVEIKTGFKLRSAVMGDNQCITVLSVFPLESSPNEQERCAEDNA
    ARVAASLAKVTSACGIFLKPDETFVHSGFIYFGPKQYLNGIQLPQSLKTAARMAPLSDAIFDDLQGTLAS
    IGTAFERSISETRHILPSRVAAAFHTYFSVRILQHHHLGFHKGSDLGQLAINKPLDFGTIALSLAVPQVL
    GGLSFLNPEKCLYRNLGDPVTSGLFQLKHYLSMVGMSDIFHALVAKSPGNCSAIDFVLNPGGLNVPGSQD
    LTSFLRQIVRRSITLSARNKLINTLFHASADLEDELVCKWLLSSTPVMSRFAADIFSRTPSGKRLQILGY
    LEGTRTLLASKMISNNAETPILERLRKITLQRWNLWFSYLDHCDSALMEAIQPIRCTVDIAQILREYSWA
    HILGGRQLIGATLPCIPEQFQTTWLKPYEQCVECSSTNNSSPYVSVALKRNVVSAWPDASRLGWTIGDGI
    PYIGSRTEDKIGQPAIKPRCPSAALREAIELTSRLTWVTQGSANSDQLIRPFLEARVNLSVQEILQMTPS
    HYSGNIVHRYNDQYSPHSFMANRMSNTATRLMVSTNTLGEFSGGGQAARDSNIIFQNVINFAVALYDIRF
    RNTCTSSIQYHRAHIHLTDCCTREVPAQYLTYTTTLNLDLSKYRNNELIYDSEPLRGGLNCNLSIDSPLM
    KGPRLNIIEDDLIRLPHLSGWELAKTVLQSIISDSSNSSTDPISSGETRSFTTHFLTYPKIGLLYSFGAL
    ISFYLGNTILCTKKIGLTEFLYYLQNQIHNLSHRSLRIFKPTFRHSSVMSRLMDIDPNFSIYIGGTAGDR
    GLSDAARLFLRIAISTFLSFVEEWVIFRKANIPLWVVYPLEGQRPDPPGEFLNRVKSLIVGIEDDKNKGS
    ILSRSEEKCSSNLVYNCKSTASNFFHASLAYWRGRHRPKKTIGATKATTAPHIILPLGNSDRPPGLDLNQ
    SNDTFIPTRIKQIVQGDSRNDRTTTTRLPPQSRSTPTSATEPPTKIYEGSTTYRGKSTDTHLDEGHNAKE
    FPFNPHRLVVPFFKLTKDGEYSIEPSPEESRSNIKGLLQHLRTMVDTTIYCRFTGIVSSMHYKLDEVLWE
    YNKFESAVTLAEGEGSGALLLIQKYGVKKLFLNTLATEHSIESEVISGYTTPRMLLSVMPRTHRGELEVI
    LNNSASQITDITHRDWFSNQKNRIPNDVDIITMDAETTENLDRSRLYEAVYTIICNHINPKTLKVVILKV
    FLSDLDGMCWINNYLAPMFGSGYLIKPITSSARSSEWYLCLSNLLSTLRTTQHQTQANCLHVVQCALQQQ
    VQRGSYWLSHLTKYTTSRLHNSYIAFGFPSLEKVLYHRYNLVDSRNGPLVSITRHLALLQTEIRELVTDY
    NQLRQSRTQTYHFIKTSKGRITKLVNDYLRFELVIRALKNNSTWHHELYLLPELIGVCHRFNHTRNCTCS
    ERFLVQTLYLHRMSDAEIKLMDRLTSLVNMFPEGFRSSSV'''

    sequence = sequence.replace('\n', '')

    features = protein_features(sequence)
