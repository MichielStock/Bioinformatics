# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 2016
Last update Thu Jun 9 2016

@author: michielstock

Optimizing a sequence using hill climbing, simulated annealing and genetic
algorithms
"""

import numpy as np
from random import choice, shuffle
from position_specific_substitution_sampling import PositionSpecificSubstitutionSampler

def mutate_peptide(sequence, characters, pmut=0.05):
    """
    Replaces each character of the sequence with an arbitrary chosen
    character with a probability pmut
    """
    mutated_sequence = ''
    for character in peptide:
        if np.random.rand() < pmut:
            mutated_sequence += choice(characters)
        else:
            mutated_sequence += character
    return mutated_sequence

def crossover_sequences(sequence1, sequence2):
    """
    Performs crossover for two sequences

    Inputs:
        - sequence1, sequence2

    Outputs:
        - crossed_sequence1, crossed_sequence2

    """
    crossed_sequence1 = ''
    crossed_sequence2 = ''
    for char1, char2 in zip(sequence1, sequence2):
        if choice([True, False]):
            crossed_sequence1 += AA2
            crossed_sequence1 += AA1
        else:
            crossed_peptide1 += AA1
            crossed_peptide2 += AA2
    return crossed_peptide1, crossed_peptide2


def tournament_selection(scored_peptides):
    """
    Apply tournament selection on a list of scored peptides.

    Input:
        - scored_peptides : a list of scored peptides, each element is a tuple
                        of the form (score, peptide)
    Output:
        - selected_peptides : a list of peptides selected from scored_peptides
                        based on tournament selection (without the score)
    """
    n_peptides = len(scored_peptides)
    selected_peptides = []
    while len(selected_peptides) < n_peptides:
        score1, peptide1  = choice(scored_peptides)
        score2, peptide2 = choice(scored_peptides)
        if score1 > score2:
            selected_peptides.append(peptide1)
        else:
            selected_peptides.append(peptide2)
    return selected_peptides


def recombinate(population, pmut, pcross):
    """
    Recombinates a population of peptides.

    Inputs:
        - population : a list of peptides
        - pmut : probability of mutating an amino acid
        - pcross : probability of two peptides crossing over

    Output:
        - recombinated_population
    """
    recombinated_population = []
    while len(population):
        if len(population) == 1:
            recombinated_population.append(population.pop())
        else:
            peptide1 = population.pop()
            peptide2 = population.pop()
            if np.random.rand < pcross:
                peptide1, peptide2 = crossover_peptides(peptide1, peptide2)
            recombinated_population += [peptide1, peptide2]
    recombinated_population = map(lambda pep : mutate_peptide(pep, pmut),
                        recombinated_population)
    return recombinated_population

def explore_peptide_region(peptide, scoring):
    """
    Yields all neighboring peptides of a given peptide that differ exactly one
    amino acid
    """
    for position in range(len(peptide)):
        for amino_acid in amino_acids:
            new_peptide = peptide[:position] + amino_acid + peptide[position+1:]
            yield (scoring(new_peptide), new_peptide)


def hill_climbing(peptidesize=None, peptide=None, scoring=score_peptide):
    """
    Uses hill climbing to find a peptide with a high score for
    antimicrobial activity.

    Inputs:
        - peptidesize : give size if stated from a randon peptide
        - peptide : optionally give an intial peptide to improve
        - scoring : the scoring function used for the peptides

    Outputs:
        - peptide : best found peptide
        - best_scores : best scores obtained through the iterations
    """

    assert peptidesize is not None or peptide is not None
    if peptide is None:
        peptide = ''
        for res in range(peptidesize):
            peptide += choice(amino_acids)
    else:
        peptidesize = len(peptide)
    best_scores = [scoring(peptide)]
    peptides = [peptide]
    while True:
        new_score, new_peptide = max(explore_peptide_region(peptide, scoring))
        if new_score > best_scores[-1]:
            best_scores.append(new_score)
            peptide = new_peptide
        else:
            break
    return peptide, best_scores


def simulated_annealing(peptidesize, Tmax, Tmin, pmut, r, kT,
                                                scoring=score_peptide):
    """
    Uses simulated annealing to find a peptide with a high score for
    antimicrobial activity.

    Inputs:
        - peptidesize : length of the peptide
        - Tmax : maximum (starting) temperature
        - Tmin : minimum (stopping) temperature
        - pmut : probability of mutating an amino acid in the peptide
        - r : rate of cooling
        - kT : number of iteration with fixed temperature
        - scoring : the scoring function used for the peptides

    Outputs:
        - peptide : best found peptide
        - fbest : best scores obtained through the iterations
        - temperatures : temperature during the iterations
    """
    # create intial peptide
    peptide = ''
    for _ in range(peptidesize):
        peptide += choice(amino_acids)

    temp = Tmax
    fstar = scoring(peptide)
    fbest = [fstar]
    temperatures = [temp]

    while temp > Tmin:
        for _ in range(kT):
            peptide_new = mutate_peptide(peptide, pmut)
            fnew = scoring(peptide_new)
            if np.exp(-(fstar - fnew) / temp) > np.random.rand():
                peptide = peptide_new
                fstar = fnew
        fbest.append(fstar)
        temp *= r
        temperatures.append(temp)
    return peptide, fbest, temperatures


def genetic_algorithm(peptidesize, n_iterations, popsize, pmut, pcross,
                                                    scoring=score_peptide):
    """
    Uses a genetic algorithm to find a peptide with a high score for
    antimicrobial activity.

    Inputs:
        - peptidesize : length of the peptide
        - n_iterations : number of iterations (generations)
        - popsize : size of the population
        - pmut : probability of mutating an amino acid in the peptide
        - pcross : probability of performing a crossover
        - scoring : the scoring function used for the peptides

    Outputs:
        - best_peptide : best found peptide
        - best_fitness_iteration : best scores obtained through the iterations
    """
    # initialize population
    population = []
    for _ in range(popsize):
        peptide = ''
        for _ in range(peptidesize):
            peptide += choice(amino_acids)
        population.append(peptide)

    # score peptides
    scored_peptides = [(scoring(peptide), peptide)
                        for peptide in population]
    best_fitness, best_peptide = max(scored_peptides)

    best_fitness_iteration = [best_fitness]

    for iter in range(n_iterations):
        # select population
        population = tournament_selection(scored_peptides)
        # recombinate population
        population = recombinate(population, pmut, pcross)
        # elitism
        population[0] = best_peptide
        # score peptides
        scored_peptides = [(scoring(peptide), peptide)
                            for peptide in population]
        # select best
        best_fitness, best_peptide = max(scored_peptides)
        best_fitness_iteration.append(best_fitness)

    return best_peptide, best_fitness_iteration


if __name__ == '__main__':
    best_peptide, best_fitness_iteration = genetic_algorithm(15, 50, 50, 0.05, 0.2)
