import numpy as np
import networkx as nx
import itertools as it
from helper import *
from tqdm import tqdm
from multiprocessing import Pool


def overlap(x, x_star):
    return np.abs((x * x_star).mean())


def compute_acceptance_proba(x, y, n, adj, a, b):
    num = 0
    denom = 0
    h = get_h(adj, a, b, n)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            denom += h[i, j] * x[i] * x[j]
            num += h[i, j] * y[i] * y[j]
    return min(1.0, np.exp(num - denom))


def compute_acceptance_proba_fast(x, v, n, adj, a, b):
    # v is the index of the changed component
    mask = np.ones(x.shape, dtype=bool)
    mask[v] = False

    h_row_masked = get_h_row(adj, a, b, n, v).reshape(-1)
    return - x[v] * np.sum(x[mask] * h_row_masked[mask])


def get_h_ij(adj, i, j, a, b, n):
    return 0.5 * (adj[i, j] * np.log(a / b) + (1 - adj[i, j]) * np.log((1 - a / n) / (1 - b / n)))


def get_h(adj, a, b, n):
    return 0.5 * (adj * np.log(a / b) + (1 - adj) * np.log((n - a) / (n - b)))


def get_h_row(adj, a, b, n, v):
    return 0.5 * (adj[v, :] * np.log(a / b) + (1 - adj[v, :]) * np.log((n - a) / (n - b)))


def metropolis_step(adj, a, b, cur_x, nb):
    comp = np.random.choice(nb, 1)
    proposed_move = cur_x.copy()
    proposed_move[comp] = -proposed_move[comp]
    #acc = compute_acceptance_proba(cur_x, proposed_move, nb, adj, a, b)
    acc = compute_acceptance_proba_fast(cur_x, comp, nb, adj, a, b)

    if np.random.uniform(0.0, 1.0) <= acc:
        return proposed_move
    else:
        return cur_x


def metropolis(adj, a, b, nb, nb_iter, x_star, *args):
    cur_x = generate_x(nb)
    overlap_list = []
    for i in range(nb_iter):
        cur_x = metropolis_step(adj, a, b, cur_x, nb)
        if x_star is not None:
            overlap_list.append(overlap(cur_x, x_star))

    return cur_x, overlap_list


def run_experiment(nb, a, b, x_star, algorithm, n0=0, nb_iter=200, nb_exp=100):
    #sum_x = np.zeros(nb)
    overlap = np.zeros((nb_exp, nb_iter))
    for j in tqdm(range(nb_exp)):
        if x_star is not None:
            adj = generate_graph(x_star, a, b)
        new_x, overlap_list = algorithm(adj, a, b, nb, nb_iter, x_star, n0)
        #sum_x += new_x
        overlap[j, :] = np.array(overlap_list)

    #x_hat = sum_x / nb_exp
    #x_hat[x_hat > 0.0] = 1
    #x_hat[x_hat <= 0.0] = -1
    return overlap.mean(axis=0)


def houdayer_step(cur_x1, cur_x2):
    y = cur_x1 * cur_x2
    diff_index = np.argwhere(y == -1)
    all_pairs = it.combinations(diff_index)
    graph_c = nx.convert.from_edgelist(all_pairs)
    rand_comp = np.random.choice(diff_index, 1)
    connected_comp = nx.algorithms.components.node_connected_component(graph_c, rand_comp)

    list_index = list(connected_comp)

    cur_x2[list_index] *= (-1)
    cur_x1[list_index] *= (-1)
    return cur_x1, cur_x2


def houdayer(adj, a, b, nb, nb_iter, x_star, *args):
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)

    overlap_list = []

    for i in range(nb_iter):
        cur_x1, cur_x2 = houdayer_step(cur_x1, cur_x2)

        cur_x2 = metropolis_step(adj, a, b, cur_x2, nb)
        cur_x1 = metropolis_step(adj, a, b, cur_x1, nb)

        if x_star is not None:
            overlap_list.append(overlap(cur_x1, x_star))

    return cur_x1, overlap_list


def mixed_houdayer(adj, a, b, nb, nb_iter_tot, x_star, n0):
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)

    overlap_list = []

    for i in nb_iter_tot:
        if i % n0 == 0:
            cur_x1, cur_x2 = houdayer_step(cur_x1, cur_x2)

        cur_x2 = metropolis_step(adj, a, b, cur_x2, nb)
        cur_x1 = metropolis_step(adj, a, b, cur_x1, nb)

        if x_star is not None:
            overlap_list.append(overlap(cur_x1, x_star))

    return cur_x1, overlap_list
