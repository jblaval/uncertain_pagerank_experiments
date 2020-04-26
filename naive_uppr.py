import numpy as np
from itertools import combinations
from standard_pagerank import PersonalizedPageRank
from graphs import UncertainGraph
from utils import generate_seed_set, get_possible_worlds
from split import simple_split


def exhPPR(graph, alpha=0.85, seed_ratio=0.5, seed_set=None):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    possible_worlds = get_possible_worlds(graph)
    pagerank_sum = np.zeros(graph.NodesNb)
    for world in possible_worlds:
        current_graph = UncertainGraph()
        current_graph.NodesNb = graph.NodesNb
        current_graph.BuildGraphFromSet(world)
        pagerank_sum += PersonalizedPageRank(
            current_graph, alpha=alpha, seed_set=seed_set
        )[1]
    return pagerank_sum / len(possible_worlds)


def B_LIN_preprocessing(graph, world, seed_set, graph_split_algo):
    world_graph = UncertainGraph()
    world_graph.NodesNb = graph.NodesNb
    world_graph.BuildGraphFromSet(world)
    current_graphs = [UncertainGraph(), UncertainGraph(), UncertainGraph()]
    nodes = graph_split_algo(world_graph) + [{0, 1}]
    for i in range(3):
        current_graphs[i].NodesNb = len(nodes[i])
    threshold = len(nodes[0])
    edges_sets = [set(), set(), set()]
    for (src, dst, e_type) in world_graph.Edges:
        if src in nodes[0] and dst in nodes[0]:
            edges_sets[0].add((src, dst, e_type))
            edges_sets[2].add((0, 0, e_type))

        elif src in nodes[1] and dst in nodes[1]:
            edges_sets[1].add((src - threshold, dst - threshold, e_type))
            edges_sets[2].add((1, 1, e_type))
        else:
            edges_sets[2].add((int(src in nodes[1]), int(dst in nodes[1]), e_type))
    seed_sets = [set(), set(), {0, 1}]
    for s in seed_set:
        if s in nodes[0]:
            seed_sets[0].add(s)
        else:
            seed_sets[1].add(s - threshold)
    for i in range(3):
        current_graphs[i].BuildGraphFromSet(edges_sets[i])
    return current_graphs, seed_sets


def B_LIN_postprocessing(current_graphs, seed_sets, pagerank_algo, alpha):
    pageranks = [
        pagerank_algo(current_graphs[i], alpha=alpha, seed_set=seed_sets[i])
        for i in range(3)
    ]
    return np.concatenate(
        [pageranks[2][1][0] * pageranks[0][1], pageranks[2][1][1] * pageranks[1][1]]
    )


def B_LIN(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    pagerank_algo=PersonalizedPageRank,
    graph_split_algo=simple_split,
):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    pagerank_sum = np.zeros(graph.NodesNb)
    possible_worlds = get_possible_worlds(graph)
    for world in get_possible_worlds(graph):
        current_graphs, seed_sets = B_LIN_preprocessing(
            graph, world, seed_set, graph_split_algo
        )
        pagerank = B_LIN_postprocessing(current_graphs, seed_sets, pagerank_algo, alpha)
        pagerank_sum += pagerank
    return pagerank_sum / len(possible_worlds)


def exhApxPPR(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    pagerank_algo=PersonalizedPageRank,
    graph_split_algo=simple_split,
):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    return B_LIN(
        graph,
        alpha=0.85,
        seed_set=seed_set,
        pagerank_algo=PersonalizedPageRank,
        graph_split_algo=simple_split,
    )


def collBase(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    UPPR_algo=exhPPR,
    UPPR_algo_kwargs=None,
):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    possible_worlds = get_possible_worlds(graph)
    sum_adjacency_matrices = np.zeros((graph.NodesNb, graph.NodesNb))
    for world in possible_worlds:
        world_graph = UncertainGraph()
        world_graph.NodesNb = graph.NodesNb
        world_graph.BuildGraphFromSet(world)
        sum_adjacency_matrices += world_graph.AdjacencyMatrix
    new_graph = UncertainGraph()
    new_graph.BuildGraphFromAdjacencyMatrix(sum_adjacency_matrices)
    kwargs = {"alpha": alpha, "seed_set": seed_set}
    if UPPR_algo_kwargs:
        kwargs.update(UPPR_algo_kwargs)
    return UPPR_algo(new_graph, **kwargs)


def collPPR(graph, alpha=0.85, seed_ratio=0.5, seed_set=None):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    return collBase(graph, alpha=alpha, seed_set=seed_set, UPPR_algo=exhPPR)


def collApxPPR(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    BLIN_pagerank_algo=PersonalizedPageRank,
    BLIN_graph_split_algo=simple_split,
):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    BLIN_kwargs = {
        "pagerank_algo": PersonalizedPageRank,
        "graph_split_algo": simple_split,
    }
    return collBase(
        graph,
        alpha=alpha,
        seed_set=seed_set,
        UPPR_algo=B_LIN,
        UPPR_algo_kwargs=BLIN_kwargs,
    )


def collApx2PPR(
    graph,
    alpha=0.85,
    seed_ratio=0.5,
    seed_set=None,
    BLIN_pagerank_algo=PersonalizedPageRank,
    BLIN_graph_split_algo=simple_split,
):
    if seed_set is None:
        seed_set = generate_seed_set(graph, seed_ratio=seed_ratio)
    possible_worlds = get_possible_worlds(graph)
    sum_adjMat = [np.zeros((len(x), len(x))) for x in BLIN_graph_split_algo(graph)] + [
        np.zeros((2, 2))
    ]
    for world in possible_worlds:
        world_graph = UncertainGraph()
        world_graph.NodesNb = graph.NodesNb
        world_graph.BuildGraphFromSet(world)
        current_graphs, seed_sets = B_LIN_preprocessing(
            graph, world, seed_set, BLIN_graph_split_algo
        )
        for i in range(3):
            sum_adjMat[i] += current_graphs[i].AdjacencyMatrix
    final_graphs = [UncertainGraph() for i in range(3)]
    for i in range(3):
        final_graphs[i].BuildGraphFromAdjacencyMatrix(sum_adjMat[i])
    return B_LIN_postprocessing(final_graphs, seed_sets, BLIN_pagerank_algo, alpha)
