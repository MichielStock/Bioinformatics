"""
Created on Sun Jan 25 2015
Last update: Tue Jan 27 2015

@author: Michiel Stock
michielfmstock@gmail.com

This file implements some functions and classes to work with
3D PDB files:
    - generate coordinates
    - plot structure in 3D
    - protein contact maps
"""


from Bio.PDB import PPBuilder
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


class StructurePDB:
    """
    Class to parse and process a PDB file
    """
    def __init__(self, structure):
        """
        Give a structure file obtained from the PDB parser
        """
        self._PDB_structure = structure
        self._residues = [residue.get_resname() for residue in
                structure.get_residues()]
        ppb = PPBuilder()  # needed to extract polypeptides in sequence
        self._sequence = ''.join(
                [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
                )

    def get_sequence(self):
        """
        Returns the AA sequence of the protein structure
        """
        return self._sequence

    def make_coordinates(self, atom_id='CA'):
        """
        Generates the 3D coordinates of an atom for each residue in the
        protein, takes CA by default or if not available in residue (i.e. glycin)
        """
        self._coordinates = []
        for model in self._PDB_structure.get_list():
            for chain in model.get_list():
                for residue in chain.get_list():
                    if residue.has_id(atom_id):
                        atom = residue[atom_id]
                        self._coordinates.append(atom.get_coord())
                    elif residue.has_id("CA"):
                        atom = residue["CA"]  # if the residue does not
                            # have the atom_id, take alpha carbon atom
                            # by default
                        self._coordinates.append(atom.get_coord())
        return self._coordinates

    def make_3D_structure_plot(self, atom_id='CA', coloring=None):
        """
        Gives a 3D plot of the residues of the protein, coloring is an
        optional dictionary to color the residues according to some property,
        takes CA by default or if not available in residue (i.e. glycin)
        """
        # first get the coordinates
        self.make_coordinates(atom_id)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x, y, z = zip(*self._coordinates)
        N = len(x)
        ax.plot(x,y,z, c = 'b')
        # determine the color map, if no dictornary for the AAs is
        # given we color them to by order
        if type(coloring) != dict:
            residue_color = plt.cm.jet(np.linspace(0,1,N))
            ax.scatter(x, y, z, c = residue_color)
        else:
            # if a dictionary mapping is given for the AA
            # use this as a coloring for the residues
            aa_mapper = lambda x:coloring[x]
            residue_color = map(aa_mapper, list(self._sequence))
            ax.scatter(x, y, z, c = residue_color, cmap='hot')
        return fig

    def make_contact_map(self, atom_id='CA', cutoff=10):
        """
        Gives a protein contact map of the residues of the protein,
        cutoff is a parameter for detemining when two residues are close
        enough for contact, takes CA by default or if not available in
        residue (i.e. glycin)
        """
        self.make_coordinates(atom_id)
        dist_matrix = distance.pdist(self._coordinates)
        # transform
        dist_matrix = distance.squareform(dist_matrix)
        fig, ax = plt.subplots()
        ax.imshow(dist_matrix<cutoff, interpolation='nearest')
        return fig, dist_matrix

    def get_interaction_matrix(self, atom_id='CA', cutoff=10):
        """
        Returns interaction matrix between residues for given cutoff
        """
        coordinates = self.make_coordinates( atom_id='CA')
        dist_matrix = distance.pdist(coordinates)
        dist_matrix = distance.squareform(dist_matrix)
        return dist_matrix <= 10


if __name__ == "__main__":
    from Bio.PDB.PDBParser import PDBParser
    parser = PDBParser()
    structure = parser.get_structure("Ebola Virus Glycoprotein", "2EBO.pdb")

    struct = StructurePDB(structure)

    # make 3D plot with standard color mapping


    # get sequence:
    sequence = struct.get_sequence()
    print sequence

    fig = struct.make_3D_structure_plot()
    fig.show()

    # use different color mapping
    # http://biopython.org/DIST/docs/api/Bio.SeqUtils.ProtParamData-pysrc.html

    from Bio.SeqUtils.ProtParam import ProtParamData

    # flexibility
    flex_map = ProtParamData.Flex
    struct.make_3D_structure_plot(coloring=flex_map).show()

    # Hydrophilicity
    hydro_map = ProtParamData.Flex
    struct.make_3D_structure_plot(coloring=hydro_map).show()

    # Make a contact map

    fig, dist_matrix = struct.make_contact_map(cutoff=10)


    fig, ax = plt.subplots()
    ax.set_xlabel('residue')
    ax.set_ylabel('residue')
    ax.set_yticks(range(222))
    ax.set_xticks(range(222))
    ax.set_xticklabels(list(sequence))
    ax.set_yticklabels(list(sequence))
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(1.5)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(1.5)
    ax.imshow(dist_matrix>11, cmap='YlGnBu', interpolation='nearest')
    ax.set_title('Protein contact map\nEbola glycoprotein')
    ax.set
    fig.savefig('contact_Ebola_glycoprotein.pdf')
