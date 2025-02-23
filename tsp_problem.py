import math
import random
from qubots.base_problem import BaseProblem
import os
import numpy as np
from collections import defaultdict


def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        return f.read().split()

class TSPProblem(BaseProblem):
    """
    Traveling Salesman Problem (TSP)

    Given a set of n cities and an (asymmetric) distance matrix between them, 
    the goal is to find a roundtrip (tour) of minimal total length visiting each city exactly once.
    
    Instance Format:
      The instance file follows the TSPLib "explicit" format. It contains a keyword "DIMENSION:" 
      followed by the number of cities and later the keyword "EDGE_WEIGHT_SECTION", after which the 
      full distance matrix is listed row by row.
    
    Decision Variables:
      A candidate solution is represented as a list of integers of length n, where the i-th element 
      is the index of the city visited in the i-th position of the tour.
    
    Objective:
      Minimize the total travel distance, computed as the sum over consecutive pairs of cities plus 
      the closing distance from the last city back to the first.
    """
    def __init__(self, instance_file):
        self.instance_file = instance_file
        tokens = read_elem(instance_file)
        token_iter = iter(tokens)
        self.nb_cities = None
        # Scan tokens to find "DIMENSION:" and then "EDGE_WEIGHT_SECTION"
        for token in token_iter:
            if token.upper() == "DIMENSION:":
                self.nb_cities = int(next(token_iter))
            if token.upper() == "EDGE_WEIGHT_SECTION":
                break
        if self.nb_cities is None:
            raise ValueError("DIMENSION not found in instance file.")
        # Read the full distance matrix (assumed to be nb_cities x nb_cities integers)
        self.dist_matrix = []
        for _ in range(self.nb_cities):
            row = [int(next(token_iter)) for _ in range(self.nb_cities)]
            self.dist_matrix.append(row)
        # (Any trailing tokens, e.g., "EOF", are ignored.)

    def evaluate_solution(self, candidate) -> float:
        """
        Evaluates a candidate tour.
        
        Parameters:
          candidate: a list of integers (city indices) representing the tour order.
          
        Returns:
          The total travel distance of the tour. If the candidate is not a valid permutation 
          of {0,...,nb_cities-1}, a heavy penalty is returned.
        """
        if sorted(candidate) != list(range(self.nb_cities)):
            return 1e9  # heavy penalty for invalid tours
        
        total_distance = 0
        n = self.nb_cities
        # Sum distance for consecutive cities.
        for i in range(1, n):
            total_distance += self.dist_matrix[candidate[i-1]][candidate[i]]
        # Add the closing leg (from last city back to first).
        total_distance += self.dist_matrix[candidate[-1]][candidate[0]]
        return total_distance

    def random_solution(self):
        """
        Generates a random candidate tour by producing a random permutation of city indices.
        """
        tour = list(range(self.nb_cities))
        random.shuffle(tour)
        return tour