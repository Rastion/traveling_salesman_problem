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

    def get_qubo(self):
        # Build the QUBO matrix as a dictionary with keys (p,q) for p <= q.
        Q = {}
        n = self.nb_cities
        dist_matrix = np.array(self.dist_matrix)
        max_distance = np.max(self.dist_matrix)
        # Set penalty parameters (A for "each city once" and B for "each position once")
        A = max_distance * n
        B = A  # we use the same penalty for both constraints

        def add_to_Q(p, q, value):
            if p > q:
                p, q = q, p
            Q[(p, q)] = Q.get((p, q), 0) + value

        # --- Objective term: tour cost ---
        # For each position t, and for each pair of cities (i, j), add:
        # d(i,j) * x[i,t] * x[j, (t+1) mod n]
        for t in range(n):
            t_next = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    p = i * n + t
                    q = j * n + t_next
                    add_to_Q(p, q, dist_matrix[i, j])

        # --- Constraint 1: Each city is visited exactly once ---
        # For each city i, add A*(sum_t x[i,t] - 1)^2.
        for i in range(n):
            for t1 in range(n):
                p = i * n + t1
                # Diagonal contribution: from -2*x + x => net -1 times A for each x.
                add_to_Q(p, p, -A)
                for t2 in range(t1 + 1, n):
                    q = i * n + t2
                    add_to_Q(p, q, 2 * A)

        # --- Constraint 2: Each position is occupied by exactly one city ---
        # For each position t, add B*(sum_i x[i,t] - 1)^2.
        for t in range(n):
            for i1 in range(n):
                p = i1 * n + t
                add_to_Q(p, p, -B)
                for i2 in range(i1 + 1, n):
                    q = i2 * n + t
                    add_to_Q(p, q, 2 * B)

        return Q

    def decode_solution(self, binary_solution):
        # --- Decode the binary vector into a tour ---
        # Reshape binary_solution into an (n x n) assignment matrix.
        n = self.nb_cities
        X = binary_solution.reshape((n, n))
        tour = []
        for t in range(n):
            col = X[:, t]
            # If exactly one city is assigned at position t, use it; otherwise take argmax.
            if np.sum(col) == 1:
                i = int(np.where(col == 1)[0][0])
            else:
                i = int(np.argmax(col))
            tour.append(i)
        # Optionally, rotate the tour so that city 0 is first.
        if 0 in tour:
            idx = tour.index(0)
            tour = tour[idx:] + tour[:idx]

        return tour
    
    def get_ising(self):
        """
        Convert the QUBO matrix to Ising parameters (h, J, offset).
        """
        Q = self.get_qubo()
        offset = 0
        # Get the number of variables from the QUBO matrix
        max_index = max(max(i, j) for i, j in Q.keys())
        n_qubits = max_index + 1  # Since indices are 0-based
        print("Number of qubits = "+str(n_qubits))
        h = defaultdict(float)
        J = defaultdict(float)
        edges = []

        for i in range(n_qubits):
            h[(i,)] -= Q.get((i, i), 0) / 2.0
            offset += Q.get((i, i), 0) / 2.0
            for j in range(i + 1, n_qubits):
                if Q.get((i, j), 0) != 0:
                    edges.append((i, j))
                J[(i, j)] += Q.get((i, j), 0) / 4.0
                h[(i,)] -= Q.get((i, j), 0) / 4.0
                h[(j,)] -= Q.get((i, j), 0) / 4.0
                offset += Q.get((i, j), 0) / 4.0

        return h, J, offset, edges