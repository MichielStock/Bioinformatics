"""
Created on Saturday 2 December 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Small-scale analysis of barstar-barnase structure
"""

from Bio.PDB import PPBuilder
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def center_mass_residue(residue):
    """
    Computes the center of mass of a residue,
    returns residue id + coordinates of center of mass
    """
    coord = 0
    mass = 0
    for atom in residue.get_atom():
        m = atom.mass
        mass += m
        coord += m * atom.coord
    return residue.resname, coord / mass

# load file
parser = PDBParser()
pdb_bb = parser.get_structure("barnstar-barnase", "1brs.pdb")

# get chains
chains = {ch.id : ch for ch in pdb_bb.get_chains()}

# compute centers of mass residues
chains_com = {id : [center_mass_residue(res) for res in ch] for id, ch in chains.items()}

split_res_com = lambda res_com : list(zip(*res_com))

res_A, coord_A = split_res_com(chains_com['A'])
res_D, coord_D = split_res_com(chains_com['D'])

coord_A = np.vstack(coord_A)
coord_D = np.vstack(coord_D)

# contact maps
dist_AA = pairwise_distances(coord_A)
dist_DD = pairwise_distances(coord_D)
dist_AD = pairwise_distances(coord_A, coord_D)

cutoff = 5
fig, (ax_AA, ax_AD, ax_DD) = plt.subplots(ncols=3)

ax_AA.imshow(dist_AA < cutoff, interpolation='nearest')
ax_AA.set_title('contactmap\nresidues barnase')
ax_AD.imshow(dist_AD < cutoff, interpolation='nearest')
ax_AD.set_title('contactmap\nresidues barnas-barstar')
ax_DD.imshow(dist_DD < cutoff, interpolation='nearest')
ax_DD.set_title('contactmap\nresidues barstar')

fig.savefig('Figures/contact_maps.png')

# 3d scatter plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wire
