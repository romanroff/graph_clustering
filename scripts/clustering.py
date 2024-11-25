from heapq import heappop as _pop, heappush as _push
from itertools import count as _count
from typing import NewType as _Nt, Union as _Union

import networkx as _nx
import logging as log
from tqdm import trange as _trange
import numpy as np
import networkx as nx
import igraph as ig

from . import utils
from tqdm import tqdm

from sklearn.cluster import HDBSCAN, KMeans
from gudhi.clustering.tomato import Tomato
import leidenalg as la

__version__ = "1.0"

__all__ = [
    "Community",
    "dijkstra_near_neighbours",
    "validate_cms",
    "resolve_louvain_communities",
    "resolve_k_means_communities"
]

Community = _Nt('Community', _Union[list[set[int]], tuple[set[int]]])


def dijkstra_near_neighbours(graph: _nx.Graph,
                             starts: list[int],
                             weight: str = 'length'):
    adjacency = graph._adj
    c = _count()
    push = _push
    pop = _pop
    dist = {}
    fringe = []
    node2cms = {
        s: i for i, s in enumerate(starts)
    }
    for start in starts:
        push(fringe, (0.0, next(c), 0, start, start))
    while fringe:
        (d, _, n, v, p) = pop(fringe)
        if v in dist:
            continue
        node2cms[v] = node2cms[p]
        dist[v] = (d, n)
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist:
                push(fringe, (vu_dist, next(c), n + 1, u, v))
    return node2cms


def validate_cms(
        graph: _nx.Graph,
        communities: Community,
        cluster_name: str = 'cluster') -> Community:
    cls = []
    for i, c in enumerate(communities):
        for n in _nx.connected_components(graph.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            graph.nodes()[j][cluster_name] = i
    return cls


# resolve_communities
def resolve_louvain_communities(H: _nx.Graph,
                                resolution: float = 1,
                                cluster_name: str = 'cluster',
                                weight: str = 'length') -> Community:
    communities = _nx.community.louvain_communities(H,
                                                    seed=1534,
                                                    weight=weight,
                                                    resolution=resolution)
    return validate_cms(H, communities, cluster_name=cluster_name)


def resolve_k_means_communities(g: _nx.Graph,
                                resolution=10,
                                max_iteration=20,
                                cluster_name: str = 'cluster',
                                weight: str = 'length',
                                print_log=False):
    communities = resolve_louvain_communities(g, resolution=resolution, cluster_name=cluster_name)
    log.info(f'communities: {len(communities)}')
    _iter = _trange(max_iteration) if print_log else range(max_iteration)
    do = True
    for _ in _iter:
        if not do:
            continue
        centers = []
        for i, cls in enumerate(communities):
            gc = g.subgraph(communities[i])
            center = _nx.barycenter(gc, weight=weight)[0]
            centers.append(center)

        node2cls = dijkstra_near_neighbours(g, centers, weight=weight)
        do = False
        for u, i in node2cls.items():
            if u not in communities[i]:
                do = True
                break
        if not do:
            continue

        communities = [set() for _ in range(len(centers))]
        for u, c in node2cls.items():
            communities[c].add(u)
        communities = validate_cms(g, communities, cluster_name=cluster_name)
    return communities
def leiden(H: nx.Graph, **kwargs) -> list[set[int]]:
    '''
    Clustering by leiden algorithm - a modification of louvain
    '''
    # Leiden works with igraph framework
    G = ig.Graph.from_networkx(H)
    # Get clustering
    partition = la.find_partition(G, **kwargs)
    # Collect corresponding nodes
    communities = []
    for community in partition:
        node_set = set()
        for v in community:
            node_set.add(G.vs[v]['_nx_name'])
        communities.append(node_set)

    return utils.validate_cms(H, communities)