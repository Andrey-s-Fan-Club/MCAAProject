import numpy as np
import networkx as nx


def generate_x(size):
    return np.random.choice(np.array([-1, 1]), size)


def generate_graph(x, a, b):
    n = len(x)
    adjacency_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            prod = x[i] * x[j]

            if prod == -1:
                value = 1 if np.random.uniform(0.0, 1.0) < b / n else 0
            else:
                value = 1 if np.random.uniform(0.0, 1.0) < a / n else 0

            adjacency_mat[i, j] = value
            adjacency_mat[j, i] = value
    return adjacency_mat
