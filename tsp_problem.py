import math
import random
from qubots.base_problem import BaseProblem
import os

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

    This class now supports both TSPLib explicit distance matrices (via EDGE_WEIGHT_SECTION)
    and coordinate-based instances (via NODE_COORD_SECTION). In the latter case, distances 
    are computed using the ATT (pseudo-Euclidean) formula.
    """
    def __init__(self, instance_file):
        self.instance_file = instance_file
        tokens = read_elem(instance_file)
        token_iter = iter(tokens)
        self.nb_cities = None
        self.node_coords = None
        self.dist_matrix = None
        edge_weight_type = None

        # Scan tokens to find key fields.
        for token in token_iter:
            upper_token = token.upper()
            if upper_token == "DIMENSION:":
                self.nb_cities = int(next(token_iter))
            elif upper_token == "EDGE_WEIGHT_TYPE:":
                edge_weight_type = next(token_iter)
            elif upper_token == "EDGE_WEIGHT_SECTION":
                # The instance provides an explicit full distance matrix.
                self.dist_matrix = []
                for _ in range(self.nb_cities):
                    row = [int(next(token_iter)) for _ in range(self.nb_cities)]
                    self.dist_matrix.append(row)
                break
            elif upper_token == "NODE_COORD_SECTION":
                # Read node coordinates: expect each line to have: index x y
                self.node_coords = []
                for _ in range(self.nb_cities):
                    # Discard the index.
                    next(token_iter)
                    x = float(next(token_iter))
                    y = float(next(token_iter))
                    self.node_coords.append((x, y))
                break

        if self.nb_cities is None:
            raise ValueError("DIMENSION not found in instance file.")

        # If we have coordinates, build the distance matrix using the ATT formula.
        if self.dist_matrix is None and self.node_coords is not None:
            self.dist_matrix = []
            for i in range(self.nb_cities):
                row = []
                xi, yi = self.node_coords[i]
                for j in range(self.nb_cities):
                    if i == j:
                        row.append(0)
                    else:
                        xj, yj = self.node_coords[j]
                        dx = xi - xj
                        dy = yi - yj
                        # Compute the pseudo-Euclidean distance per TSPLib ATT rules.
                        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                        tij = int(rij)
                        dij = tij + 1 if tij < rij else tij
                        row.append(dij)
                self.dist_matrix.append(row)

    def evaluate_solution(self, candidate) -> float:
        if sorted(candidate) != list(range(self.nb_cities)):
            return 1e9  # heavy penalty for invalid tours
        total_distance = 0
        n = self.nb_cities
        for i in range(1, n):
            total_distance += self.dist_matrix[candidate[i-1]][candidate[i]]
        total_distance += self.dist_matrix[candidate[-1]][candidate[0]]
        return total_distance

    def random_solution(self):
        tour = list(range(self.nb_cities))
        random.shuffle(tour)
        return tour
