import numpy as np
from utils import generate_seed_set
from copy import deepcopy


def PersonalizedPageRank(graph, alpha=0.85, seed_ratio=0.5, seed_set=None):

    n = graph.NodesNb
    if n == 0:
        raise Exception("Cannot compute PageRank on an empty graph.")
    graph.NormalizedAdjacencyMatrix = deepcopy(graph.AdjacencyMatrix)
    for null_row in np.where(~graph.NormalizedAdjacencyMatrix.any(axis=1)):
        graph.NormalizedAdjacencyMatrix[null_row] = np.full(n, 1 / n)
    row_sums = graph.NormalizedAdjacencyMatrix.sum(axis=1).reshape(-1, 1)
    graph.NormalizedAdjacencyMatrix /= row_sums

    seed_matrix = np.zeros((n, n))
    if seed_set is None:
        nb_seed_pages = int(n * seed_ratio)
        seed_set = generate_seed_set(graph, nb_seed=nb_seed_pages)
    else:
        nb_seed_pages = len(seed_set)
        seed_set = np.sort(list(seed_set))
    for p in seed_set:
        seed_matrix[:, p] = 1 / nb_seed_pages

    eigen = np.linalg.eig(
        (
            alpha * graph.NormalizedAdjacencyMatrix + (1 - alpha) * seed_matrix
        ).transpose()
    )
    first_eigenvect = eigen[1][:, np.argmax(np.absolute(eigen[0]))]
    assert np.linalg.norm(np.imag(first_eigenvect)) == 0
    pagerank = np.real(first_eigenvect)
    pagerank /= np.sum(pagerank)

    return (seed_set, pagerank)


def StdPageRank(graph, alpha=0.85):

    return PersonalizedPageRank(graph, alpha=alpha, seed_ratio=1)
