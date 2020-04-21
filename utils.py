import numpy as np
from itertools import combinations
from graphs import UncertainGraph


def simple_split(graph):
    half = graph.NodesNb / 2
    return [set(range(int(half))), set(range(int(half), graph.NodesNb))]


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
