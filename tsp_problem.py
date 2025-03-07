import math
from qubots.base_problem import BaseProblem
import os

def parse_explicit_matrix(tokens, nb_nodes, weight_format="FULL_MATRIX"):
    # Remove any non-numeric tokens such as "EOF"
    tokens = [token for token in tokens if token.upper() != "EOF"]
    weight_format = weight_format.upper().replace("_", " ").strip()
    matrix = [[0] * nb_nodes for _ in range(nb_nodes)]
    
    if weight_format == "FUNCTION":
        raise NotImplementedError("FUNCTION weight type is not implemented")
    
    elif weight_format in ["FULL MATRIX"]:
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                matrix[i][j] = int(next(it))
                
    elif weight_format == "UPPER ROW":
        # Upper triangular (row-wise) without diagonal: tokens for i < j.
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes - 1):
            for j in range(i + 1, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "LOWER ROW":
        # Lower triangular (row-wise) without diagonal: tokens for i > j.
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(1, nb_nodes):
            for j in range(i):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "UPPER DIAG ROW":
        # Upper triangular (row-wise) including diagonal: tokens for i <= j.
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER DIAG ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(i, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "LOWER DIAG ROW":
        # Lower triangular (row-wise) including diagonal: tokens for i >= j.
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER DIAG ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(i + 1):  # j = 0 to i
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "UPPER COL":
        # Upper triangular (column-wise) without diagonal:
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 1 to n-1, rows i=0..(j-1)
        for j in range(1, nb_nodes):
            for i in range(j):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "LOWER COL":
        # Lower triangular (column-wise) without diagonal:
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-2, rows i=j+1..(n-1)
        for j in range(nb_nodes - 1):
            for i in range(j + 1, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "UPPER DIAG COL":
        # Upper triangular (column-wise) including diagonal:
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER DIAG COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-1, rows i=0..j
        for j in range(nb_nodes):
            for i in range(j + 1):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "LOWER DIAG COL":
        # Lower triangular (column-wise) including diagonal:
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER DIAG COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-1, rows i from j to n-1
        for j in range(nb_nodes):
            for i in range(j, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value
    else:
        raise ValueError("Unsupported EDGE_WEIGHT_FORMAT: " + weight_format)
    
    return matrix


def parse_coordinates(tokens, nb_cities):
    """
    Parses coordinates from a NODE_COORD_SECTION.
    Each line is expected to have: <node_index> <x> <y>
    """
    coords = []
    it = iter(tokens)
    for _ in range(nb_cities):
        next(it)  # Skip the node index
        x = float(next(it))
        y = float(next(it))
        coords.append((x, y))
    return coords

def compute_distance_matrix(coords, edge_weight_type, node_coord_type):
    """
    Builds the full distance matrix by applying the correct distance function
    for each pair of nodes.
    """
    nb = len(coords)
    matrix = [[0] * nb for _ in range(nb)]
    for i in range(nb):
        for j in range(nb):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = calc_distance(coords[i], coords[j], edge_weight_type, node_coord_type)
    return matrix

def calc_distance(c1, c2, edge_weight_type, node_coord_type):
    """
    Computes the distance between two nodes c1 and c2 based on the EDGE_WEIGHT_TYPE.
    The node_coord_type can be used to decide between 2D and 3D calculations.
    """
    etype = edge_weight_type.upper()
    
    if etype in ["EUC_2D", "EUC 2D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return int(round(math.sqrt(dx * dx + dy * dy)))
    
    elif etype in ["EUC_3D", "EUC 3D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dz = c1[2] - c2[2] if len(c1) > 2 and len(c2) > 2 else 0
        return int(round(math.sqrt(dx * dx + dy * dy + dz * dz)))
    
    elif etype in ["MAN_2D", "MAN 2D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        return int(round(dx + dy))
    
    elif etype in ["MAN_3D", "MAN 3D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        dz = abs(c1[2] - c2[2]) if len(c1) > 2 and len(c2) > 2 else 0
        return int(round(dx + dy + dz))
    
    elif etype in ["MAX_2D", "MAX 2D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        return max(int(round(dx)), int(round(dy)))
    
    elif etype in ["MAX_3D", "MAX 3D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        dz = abs(c1[2] - c2[2]) if len(c1) > 2 and len(c2) > 2 else 0
        return max(int(round(dx)), int(round(dy)), int(round(dz)))
    
    elif etype == "GEO":
        # For GEO, the coordinates are in DDD.MM format.
        def to_radians(coord):
            deg = int(coord)
            minutes = coord - deg
            return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0

        lat1 = to_radians(c1[0])
        lon1 = to_radians(c1[1])
        lat2 = to_radians(c2[0])
        lon2 = to_radians(c2[1])
        RRR = 6378.388
        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)
    
    elif etype == "ATT":
        # Pseudo-Euclidean (ATT) distance.
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
        tij = int(round(rij))
        return tij + 1 if tij < rij else tij
    
    elif etype in ["CEIL_2D", "CEIL 2D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return int(math.ceil(math.sqrt(dx * dx + dy * dy)))
    
    elif etype in ["XRAY1", "XRAY2"]:
        # Special distance functions for crystallography problems.
        # A proper implementation would mimic the original subroutine.
        # Here we provide a placeholder (e.g. scaling Euclidean distance).
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        d = math.sqrt(dx * dx + dy * dy)
        return int(round(d * 100))
    
    else:
        raise ValueError("Unsupported EDGE_WEIGHT_TYPE: " + edge_weight_type)
    



def read_header_and_data(instance_file):
    """
    Reads the TSPLIB file line-by-line, extracting header key/value pairs until
    a known data section is encountered (e.g. NODE_COORD_SECTION or EDGE_WEIGHT_SECTION).
    Returns a tuple (header, section, data_lines) where:
    - header is a dict mapping header keys (uppercase) to values,
    - section is the name of the data section encountered, and
    - data_lines is a list of strings containing the rest of the file.
    """
    header = {}
    data_lines = []
    section = None

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(instance_file):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        instance_file = os.path.join(base_dir, instance_file)

    with open(instance_file) as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines.
            if not stripped:
                continue

            upper_line = stripped.upper()
            # Check if we have reached a data section.
            if upper_line in ["NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION",
                            "DISPLAY_DATA_SECTION", "DEPOT_SECTION"]:
                section = upper_line
                # The rest of the file belongs to the data section.
                # We add the current line's content if needed or simply break.
                # In many TSPLIB files, the section header itself is not part of the data.
                break

            # Process header line: expect "KEY : VALUE" format.
            if ':' in stripped:
                key, value = stripped.split(":", 1)
                header[key.strip().upper()] = value.strip()
    
        # Read remaining lines (data part)
        if section:
            for line in f:
                if line.strip():  # ignore empty lines
                    data_lines.append(line.strip())
    
    return header, section, data_lines

class TSPProblem(BaseProblem):
    """
    Traveling Salesman Problem (TSP)

    This class now supports both TSPLib explicit distance matrices (via EDGE_WEIGHT_SECTION)
    and coordinate-based instances (via NODE_COORD_SECTION). In the latter case, distances 
    are computed using the ATT (pseudo-Euclidean) formula.
    """
    def __init__(self, instance_file):
        header, section, data_lines = read_header_and_data(instance_file)
        print("Header:", header)  # For debugging

        # Set header fields with defaults if missing.
        self.name = header.get("NAME", "Unknown")
        self.comment = header.get("COMMENT")
        self.problem_type = header.get("TYPE", "TSP")
        self.nb_cities = int(header.get("DIMENSION"))
        self.edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EXPLICIT")
        self.edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
        self.node_coord_type = header.get("NODE_COORD_TYPE", "NO_COORDS")
        
        # Tokenize the data section lines.
        tokens = []
        for line in data_lines:
            tokens.extend(line.split())
        
        if section == "EDGE_WEIGHT_SECTION":
            self.dist_matrix = parse_explicit_matrix(tokens, self.nb_cities, self.edge_weight_format)
        elif section == "NODE_COORD_SECTION":
            self.coords = parse_coordinates(tokens, self.nb_cities)
            self.dist_matrix = compute_distance_matrix(self.coords, self.edge_weight_type, self.node_coord_type)
        else:
            raise ValueError("Unsupported or missing data section in instance file.")

    def evaluate_solution(self, candidate):
        if sorted(candidate) != list(range(self.nb_cities)):
            return float('inf')
        total = 0
        for i in range(1, self.nb_cities):
            total += self.dist_matrix[candidate[i - 1]][candidate[i]]
        total += self.dist_matrix[candidate[-1]][candidate[0]]
        return total

    def random_solution(self):
        import random
        tour = list(range(self.nb_cities))
        random.shuffle(tour)
        return tour