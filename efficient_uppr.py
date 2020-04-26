import numpy as np
from graphs import StdGraph, UncertainGraph
from split import generalized_kernighan_lin
from utils import generate_seed_set, get_possible_worlds, inverse_block_diagonal_matrix
from copy import deepcopy


def decompose(
    graph,
    split_algo=generalized_kernighan_lin,
    split_kwargs={"accuracy": 30, "nb_part": 2, "nb_iterations": 10},
):
    n = graph.NodesNb
    uncertain_graph = UncertainGraph()
    certain_graph = StdGraph()
    uncertain_graph.NodesNb = n
    certain_graph.NodesNb = n
    edges_sets = [set(), set()]
    for edge in graph.Edges:
        edges_sets[int(edge[2] == 1)].add(edge)
    uncertain_graph.BuildGraphFromSet(edges_sets[0])
    certain_graph.BuildGraphFromSet(edges_sets[1])
    partition = [list(part) for part in split_algo(certain_graph, **split_kwargs)]
    permutation = [x for part in partition for x in part]
    graph.ChangeBase(permutation)
    uncertain_graph.ChangeBase(permutation)
    certain_graph.ChangeBase(permutation)
    part_assign = []
    for i in range(len(partition)):
        part_assign += len(partition[i]) * [i]
    T_BL = StdGraph()
    T_X = StdGraph()
    T_BL.NodesNb = n
    T_X.NodesNb = n
    T_BL_edges = set()
    T_X_edges = set()
    for src, dst, _ in graph.Edges:
        if part_assign[src] == part_assign[dst]:
            T_BL_edges.add((src, dst, 1))
        else:
            T_X_edges.add((src, dst, 1))
    T_BL.BuildGraphFromSet(T_BL_edges)
    T_X.BuildGraphFromSet(T_X_edges)
    P = []
    for world in get_possible_worlds(uncertain_graph):
        current_graph = UncertainGraph()
        current_graph.NodesNb = n
        current_graph.BuildGraphFromSet(world)
        P.append(current_graph)
    return T_BL, T_X, P, permutation, partition


def efficient_UPPR(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    split_algo=generalized_kernighan_lin,
    split_kwargs={"accuracy": 30, "nb_part": 2, "nb_iterations": 10},
):
    print(graph.AdjacencyMatrix)
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    seed_length = len(seed_set)
    n = graph.NodesNb
    graph_T_BL, graph_T_X, graph_list_P, permutation, partition = decompose(
        graph, split_algo=split_algo, split_kwargs=split_kwargs
    )
    T_BL = graph_T_BL.AdjacencyMatrix
    T_X = graph_T_X.AdjacencyMatrix
    normalization_matrix = T_BL + T_X
    P = [g.AdjacencyMatrix for g in graph_list_P]
    for world in P:
        for null_row in np.where(~(T_BL + T_X + world).any(axis=1)):
            world[null_row] = np.full(
                n, 1 / n
                )
        normalization_matrix += np.dot(1/n, world)
    row_sums = normalization_matrix.sum(axis=1).reshape(-1, 1)
    for matrix in [T_BL, T_X] + P:
        matrix = np.transpose(matrix / row_sums)
    Q = np.identity(n) - alpha * T_BL
    Q_inv = inverse_block_diagonal_matrix(Q, [len(part) for part in partition])
    uncertain_sum = np.zeros((n, n))
    for world in P:
        uncertain_sum += world
    seed_vector = np.array(
        [[1 / seed_length] if i in seed_set else [0] for i in range(n)]
    )
    permuted_pagerank = np.dot(
        1 - alpha,
        np.linalg.multi_dot(
            [
                (
                    np.identity(n)
                    + np.dot(
                        alpha / n,
                        np.linalg.multi_dot(
                            [
                                Q_inv,
                                (
                                    (np.dot(n, T_X) + uncertain_sum)
                                    + np.dot(
                                        alpha,
                                        (
                                            np.dot(
                                                n,
                                                np.linalg.multi_dot([T_X, Q_inv, T_X]),
                                            )
                                            + np.linalg.multi_dot(
                                                [uncertain_sum, Q_inv, T_X]
                                            )
                                            + np.linalg.multi_dot(
                                                [T_X, Q_inv, uncertain_sum]
                                            )
                                        ),
                                    )
                                ),
                            ]
                        ),
                    )
                ),
                Q_inv,
                seed_vector,
            ]
        ),
    )
    return np.array([permuted_pagerank[permutation.index(i)] for i in range(n)])


if __name__ == "__main__":
    G = UncertainGraph()
    G.BuildGraphFromTxt("data/test.txt")
    print(efficient_UPPR(G))
