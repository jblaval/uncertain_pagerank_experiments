import numpy as np
from itertools import combinations
from graphs import UncertainGraph


def generate_seed_set(graph, seed_ratio=None, nb_seed=None):
    if nb_seed:
        pass
    else:
        return set(
            np.random.choice(
                graph.NodesNb, int(graph.NodesNb * seed_ratio), replace=False
            )
        )


def get_possible_worlds(graph):
    certain_edges = set()
    uncertain_edges = set()
    for edge in graph.Edges:
        certain_edges.add(edge) if edge[-1] == 1 else uncertain_edges.add(edge)
    possible_uncertain_worlds = []
    for size in range(len(uncertain_edges) + 1):
        possible_uncertain_worlds += combinations(uncertain_edges, size)
    possible_worlds = []
    for world in possible_uncertain_worlds:
        possible_worlds.append(set(world) | certain_edges)
    return possible_worlds


def inverse_block_diagonal_matrix(matrix, block_dimensions):
    n = len(matrix)
    assert sum(block_dimensions) == n
    inversed_matrix = np.zeros((n, n))
    i = 0
    block = 0
    while block < len(block_dimensions):
        current_block = matrix[
            i : i + block_dimensions[block], i : i + block_dimensions[block]
        ]
        inversed_matrix[
            i : i + block_dimensions[block], i : i + block_dimensions[block]
        ] = current_block
        i += block_dimensions[block]
        block += 1
    return inversed_matrix


def distance(vector1, vector2):
    return np.linalg.norm(vector2 - vector1)
