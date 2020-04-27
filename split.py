import numpy as np
from copy import deepcopy
from graphs import UncertainGraph


def simple_split(graph):
    half = graph.NodesNb / 2
    return [set(range(int(half))), set(range(int(half), graph.NodesNb))]


def kernighan_lin(graph, accuracy=30, nb_iterations=10):
    best_cost = np.Infinity
    n = graph.NodesNb
    for _ in range(nb_iterations):
        randomized_part = np.random.choice(n, int(n / 2), replace=False)
        P = [int(i in randomized_part) for i in range(n)]
        current_best_P = deepcopy(P)
        cost = np.Infinity
        for j in range(accuracy + 1):
            gain = np.zeros(n)
            for (src, dst, e_type) in graph.Edges:
                current_gain = e_type * (int(P[src] != P[dst]) - 0.5) * 2
                gain[src] += current_gain
                gain[dst] += current_gain
            current_cost = sum([max(0, g) for g in gain])
            if current_cost < cost:
                current_best_P = deepcopy(P)
                cost = current_cost
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_p = deepcopy(current_best_P)
                break
            if j == accuracy:
                print(
                    f"Kernighan-Lin algorithm has ended after {accuracy} iterations. You can modify the accuracy parameter to get further."
                )
                if cost < best_cost:
                    best_cost = cost
                    best_p = deepcopy(current_best_P)
                break
            while len(np.where(gain >= 0)[0]) > 1:
                best0, best1 = [
                    np.argmax(
                        [gain[i] if P[i] == j else -np.Infinity for i in range(n)]
                    )
                    for j in range(2)
                ]
                if best0 + best1 < 0:
                    break
                for node in best0, best1:
                    gain[node] = -np.Infinity
                    for i in range(n):
                        for way in (
                            graph.AdjacencyMatrix[i, node],
                            graph.AdjacencyMatrix[node, i],
                        ):
                            if way != 0:
                                gain[i] -= way * (int(P[i] != P[node]) - 0.5) * 2
                mem = P[best0]
                P[best0] = P[best1]
                P[best1] = mem
    return [set(np.where(np.array(best_p) == i)[0]) for i in range(2)]


def generalized_kernighan_lin(graph, accuracy=30, nb_part=2, nb_iterations=10):
    assert nb_part >= 2
    assert nb_part <= graph.NodesNb
    if nb_part == 2:
        return kernighan_lin(graph, accuracy=accuracy)
    else:
        bipart = [
            list(x)
            for x in kernighan_lin(
                graph, accuracy=accuracy, nb_iterations=nb_iterations
            )
        ]
        edge_lists = ([], [])
        for src, dst, e_type in graph.Edges:
            if src in bipart[0] and dst in bipart[0]:
                edge_lists[0].append(
                    (bipart[0].index(src), bipart[0].index(dst), e_type)
                )
            elif src in bipart[1] and dst in bipart[1]:
                edge_lists[1].append(
                    (bipart[1].index(src), bipart[1].index(dst), e_type)
                )
        graphs = (UncertainGraph(), UncertainGraph())
        for i in range(2):
            graphs[i].NodesNb = len(bipart[i])
            graphs[i].BuildGraphFromSet(edge_lists[i])
        return [
            {bipart[0][idx] for idx in part_set}
            for part_set in generalized_kernighan_lin(
                graphs[0],
                accuracy=accuracy,
                nb_part=int(nb_part / 2),
                nb_iterations=nb_iterations,
            )
        ] + [
            {bipart[1][idx] for idx in part_set}
            for part_set in generalized_kernighan_lin(
                graphs[1],
                accuracy=accuracy,
                nb_part=int(nb_part / 2),
                nb_iterations=nb_iterations,
            )
        ]
