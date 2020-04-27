import numpy as np
import csv

from time import perf_counter

from graphs import UncertainGraph, StdGraph
from standard_pagerank import StdPageRank, PersonalizedPageRank
from split import simple_split, generalized_kernighan_lin
from utils import generate_seed_set, distance
from naive_uppr import (
    exhPPR,
    exhApxPPR,
    collPPR,
    collApxPPR,
    collApx2PPR,
)

if __name__ == "__main__":    

    # alpha tests

    alpha_results = []
    G = UncertainGraph()
    G.BuildGraphFromTxt(f"data/facebook_combined.txt", uncertainty=0.5)
    seed_set = generate_seed_set(G, seed_ratio = 0.75)
    for a in range(1, 6):
        t0 = perf_counter()
        pagerank = exhPPR(G, seed_set = seed_set, alpha = 1/a)
        duration = perf_counter() - t0
        alpha_results.append([1/a, duration, np.var(pagerank)])
    
    with open('results/alpha.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'duration', 'variance'])
        for row in alpha_results:
            writer.writerow(row)


    # seed ratio tests

    seed_results = []
    G = UncertainGraph()
    G.BuildGraphFromTxt(f"data/facebook_combined.txt", uncertainty=0.5)
    for s in range(1, 6):
        t0 = perf_counter()
        pagerank = exhPPR(G, seed_ratio = 1/s)
        duration = perf_counter() - t0
        seed_results.append([1/s, duration, np.var(pagerank)])
    
    with open('results/seed_ratio.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['seed_ratio', 'duration', 'variance'])
        for row in seed_results:
            writer.writerow(row)


    # uncertainty tests

    uncertainty_results = []
    G = UncertainGraph()
    G.BuildGraphFromTxt(f"data/facebook_combined.txt")
    seed_set = generate_seed_set(G, seed_ratio = 0.75)
    for u in range(1, 6):
        G = UncertainGraph()
        G.BuildGraphFromTxt(f"data/facebook_combined.txt", uncertainty=1/u)
        t0 = perf_counter()
        pagerank = exhPPR(G, seed_set = seed_set)
        duration = perf_counter() - t0
        seed_results.append([1/u, duration, np.var(pagerank)])
    
    with open('results/uncertainty.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['uncertainty', 'duration', 'variance'])
        for row in uncertainty_results:
            writer.writerow(row)


    # size tests

    size_results = []
    for i in range(10):
        G = UncertainGraph()
        G.BuildGraphFromTxt(f"data/fc{i}.txt", uncertainty=0.5)
        t0 = perf_counter()
        pagerank = exhPPR(G, seed_set = 0.75)
        duration = perf_counter() - t0
        size_results.append([f'{i*10}%', duration])
    
    with open('results/alpha.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'duration'])
        for row in alpha_results:
            writer.writerow(row)

    # algos tests

    algos_results = []
    G = UncertainGraph()
    G.BuildGraphFromTxt(f"data/facebook_combined.txt", uncertainty=0.5)
    seed_set = generate_seed_set(G, seed_ratio = 0.75)
    exact_pagerank = exhPPR(G, seed_set = seed_set)
    for algo in [exhApxPPR, collPPR, collApxPPR, collApx2PPR]:
        t0 = perf_counter()
        pagerank = algo(G, seed_set = seed_set, graph_split_algo=simple_split)
        duration = perf_counter() - t0
        seed_results.append((duration, distance(pagerank, exact_pagerank)))
        t0 = perf_counter()
        pagerank = algo(G, seed_set = seed_set, graph_split_algo=generalized_kernighan_lin)
        duration = perf_counter() - t0
        seed_results.append([str(algo), duration, distance(pagerank, exact_pagerank)])
    
    with open('results/algos.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['algos', 'duration', 'l2_error'])
        for row in algos_results:
            writer.writerow(row)
    

    # number of subdivisions tests

    np_sub_results = []
    G = UncertainGraph()
    G.BuildGraphFromTxt(f"data/facebook_combined.txt")
    seed_set = generate_seed_set(G, seed_ratio = 0.75)
    exact_pagerank = exhPPR(G, seed_set = seed_set)
    for nb_part in [2, 4, 8, 16, 32]:
        t0 = perf_counter()
        pagerank = algo(G, seed_set = seed_set, graph_split_algo=lambda x: generalized_kernighan_lin(G, nb_part = nb_part))
        duration = perf_counter() - t0
        np_sub_results.append([nb_part, duration, distance(pagerank, exact_pagerank)])
    
    with open('results/nb_parts.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['nb_parts', 'duration', 'l2_error'])
        for row in alpha_results:
            writer.writerow(row)