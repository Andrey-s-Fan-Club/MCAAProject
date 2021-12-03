import numpy as np
import networkx as nx
import itertools as it
from graph_helper import *


def compute_acceptance_proba(x, y, n, adj, a, b):
    num = 0
    denom = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            hij = get_h_ij(adj, i, j, a, b, n)
            num += hij * x[i] * x[j]
            denom += hij * y[i] * y[j]
    return np.min(1, np.exp(num - denom))


def get_h_ij(adj, i, j, a, b, n):
    return 1 / 2 * (adj[i, j] * np.log(a / b) + (1 - adj[i, j]) * np.log((1 - a / n) / (1 - b / n)))


def metropolis_step(adj, a, b, cur_x, nb):
    comp = np.random.choice(nb, 1)
    proposed_move = cur_x.copy()
    proposed_move[comp] = -proposed_move[comp]
    acc = compute_acceptance_proba(cur_x, proposed_move, nb, adj, a, b)

    if np.random.uniform(0.0, 1.0) < acc:
        return proposed_move
    else:
        return cur_x


def metropolis(adj, a, b, nb, nb_iter):
    cur_x = generate_x(nb)
    for i in range(nb_iter):
        cur_x = metropolis_step(adj, a, b, cur_x, nb)

    return cur_x


def run_experiment(adj, nb, a, b, nb_iter=200, nb_exp=100):
    sum_x = np.zeros(nb)
    for j in range(nb_exp):
        sum_x += metropolis(adj, a, b, nb, nb_iter)
    x_hat = sum_x / nb_exp
    x_hat[x_hat > 0.0] = 1
    x_hat[x_hat <= 0.0] = -1
    return x_hat


def houdayer(adj, a, b, nb, nb_iter):
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)

    for i in range(nb_iter):
        y = cur_x1 * cur_x2
        diff_index = np.argwhere(y == -1)
        all_pairs = it.combinations(diff_index)
        graph_c = nx.convert.from_edgelist(all_pairs)
        rand_comp = np.random.choice(diff_index, 1)
        connected_comp = nx.algorithms.components.node_connected_component(graph_c, rand_comp)

        list_index = list(connected_comp)

        cur_x2[list_index] *= (-1)
        cur_x1[list_index] *= (-1)

        cur_x2 = metropolis_step(adj, a, b, cur_x2, nb)
        cur_x1 = metropolis_step(adj, a, b, cur_x2, nb)

    return


def mixed_houdayer():
    return
