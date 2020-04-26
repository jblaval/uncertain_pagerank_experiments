import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from multipledispatch import dispatch


class UncertainGraph:
    def __init__(self):
        self.NodesNb = 0
        self.EdgesNb = 0
        self.Edges = set()
        self.AdjacencyMatrix = np.empty(0)

    def AddNode(self, node):
        self.NodesNb = max(self.NodesNb, node + 1)

    def AddEdge(self, src, dst, edge_type):
        self.Edges.add((src, dst, edge_type))
        self.EdgesNb += 1

    def BuildGraphFromTxt(self, data_path, uncertainty=0.5):
        with open(data_path) as f:
            for row in f:
                nodes = [int(x) for x in row.split()]
                for n in nodes:
                    if n >= self.NodesNb:
                        self.AddNode(n)
                np.random.shuffle(nodes)
                self.AddEdge(*nodes, int(np.random.uniform() > uncertainty) / 2 + 0.5)
        self.BuildAdjacencyMatrix()

    def BuildGraphFromSet(self, edges_set):
        for edge in edges_set:
            nodes = edge[:-1]
            for n in nodes:
                if n >= self.NodesNb:
                    self.AddNode(n)
            self.AddEdge(*edge)
        self.BuildAdjacencyMatrix()

    def BuildGraphFromAdjacencyMatrix(self, adjacency_matrix):
        self.AdjacencyMatrix = adjacency_matrix
        self.NodesNb = len(adjacency_matrix)
        for i in range(self.NodesNb):
            for j in range(self.NodesNb):
                if adjacency_matrix[i, j]:
                    for _ in range(int(round(adjacency_matrix[i, j]))):
                        self.AddEdge(i, j, 1)

    def BuildAdjacencyMatrix(self):
        self.AdjacencyMatrix = np.zeros((self.NodesNb, self.NodesNb))
        for edge in self.Edges:
            src, dst, edge_type = edge
            self.AdjacencyMatrix[src, dst] += edge_type

    def ChangeBase(self, permutation_list):
        assert len(permutation_list) == self.NodesNb
        newAdjacencyMatrix = np.zeros((self.NodesNb, self.NodesNb))
        for i in range(self.NodesNb):
            for j in range(self.NodesNb):
                newAdjacencyMatrix[i][j] = self.AdjacencyMatrix[permutation_list[i]][
                    permutation_list[j]
                ]
        self.AdjacencyMatrix = newAdjacencyMatrix
        self.Edges = set()
        self.EdgesNb = 0
        self.BuildGraphFromAdjacencyMatrix(self.AdjacencyMatrix)

    def Display(self, with_labels=True, pagerank=False):
        nx.draw_networkx(
            nx.DiGraph(self.AdjacencyMatrix), with_labels=with_labels, arrows=True
        )
        plt.show()


class StdGraph(UncertainGraph):
    def __init__(self):
        super().__init__()

    def BuildGraphFromTxt(self, data_path):
        super().BuildGraphFromTxt(data_path, uncertainty=0)
