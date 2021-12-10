import numpy as np
import networkx as nx
import itertools as it
from helper import *
from tqdm import tqdm
from multiprocessing import Pool


def overlap(x, x_star):
    return np.abs((x * x_star).mean())


def compute_acceptance_proba_fast(x, v, n, adj, a, b):
    # v is the index of the changed component
    mask = np.ones(x.shape, dtype=bool)
    mask[v] = False

    h_row_masked = get_h_row(adj, a, b, n, v).reshape(-1)
    return - x[v] * np.sum(x[mask] * h_row_masked[mask])


def get_h_row(adj, a, b, n, v):
    return 0.5 * (adj[v, :] * np.log(a / b) + (1 - adj[v, :]) * np.log((n - a) / (n - b)))


def metropolis_step(adj, a, b, cur_x, nb):
    comp = np.random.choice(nb, 1)
    proposed_move = cur_x.copy()
    proposed_move[comp] = -proposed_move[comp]

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
    overlap = np.zeros((nb_exp, nb_iter))
    for j in tqdm(range(nb_exp)):
        if x_star is not None:
            adj = generate_graph(x_star, a, b)

        new_x, overlap_list = algorithm(adj, a, b, nb, nb_iter, x_star, n0)
        overlap[j, :] = np.array(overlap_list)

    return overlap.mean(axis=0)


def overlap_as_function_of_r(x_star, algorithm, nb, d=3):
    range_r = np.logspace(-4, 0, 60)
    print(range_r)
    overlap_r = []
    for r in range_r:
        a = 2*d/(r+1)
        b = r*a
        overlaps = run_experiment(len(x_star), a,b,x_star, algorithm, nb_iter=3000, nb_exp=30)
        overlap_r.append(overlaps[-1])

    ax = sns.lineplot(x=range_r, y=overlap_r)
    plt.suptitle(f"Average overlap over {10} experiments")
    ax.set_xlabel("Value of r")
    ax.set_ylabel("Overlap")
    plt.show()


def houdayer_step(cur_x1, cur_x2, adj):
    y = cur_x1 * cur_x2
    diff_index = np.argwhere(y == -1).reshape(-1).tolist()
    same_index = np.argwhere(y == 1).reshape(-1).tolist()
    rand_comp = np.random.choice(diff_index, 1)[0]

    observed_graph = nx.convert_matrix.from_numpy_matrix(adj)
    observed_graph.remove_nodes_from(same_index)
    connected_comp = nx.algorithms.components.node_connected_component(observed_graph, rand_comp)

    list_index = list(connected_comp)

    cur_x2[list_index] *= (-1)
    cur_x1[list_index] *= (-1)
    return cur_x1, cur_x2


def houdayer(adj, a, b, nb, nb_iter, x_star, *args):
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)

    overlap_list = []

    for i in range(nb_iter):
        cur_x1, cur_x2 = houdayer_step(cur_x1, cur_x2, adj)

        cur_x1 = metropolis_step(adj, a, b, cur_x1, nb)
        cur_x2 = metropolis_step(adj, a, b, cur_x2, nb)

        if x_star is not None:
            overlap_list.append(overlap(cur_x1, x_star))

    return cur_x1, overlap_list



def mixed_houdayer(adj, a, b, nb, nb_iter, x_star, n0):
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)

    overlap_list = []

    for i in range(nb_iter):
        if i % n0 == 0:
            cur_x1, cur_x2 = houdayer_step(cur_x1, cur_x2, adj)

        cur_x2 = metropolis_step(adj, a, b, cur_x2, nb)
        cur_x1 = metropolis_step(adj, a, b, cur_x1, nb)

        if x_star is not None:
            overlap_list.append(overlap(cur_x1, x_star))

    return cur_x1, overlap_list
