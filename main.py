from graphs import UncertainGraph, StdGraph
from pagerank import StdPageRank, PersonalizedPageRank
from uncertain_pagerank import (
    exhPPR,
    B_LIN,
    collPPR,
    collApxPPR,
    exhApxPPR,
    collApx2PPR,
)

if __name__ == "__main__":

    G = UncertainGraph()
    G.BuildGraphFromTxt("data/test.txt")
    print(G.AdjacencyMatrix)
    pr = collApx2PPR(G)
    print(pr)
