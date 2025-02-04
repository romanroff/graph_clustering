import time

import networkx as nx
import numpy as np
from tqdm import tqdm

from common import GraphLayer
from graph_generator import extract_cluster_list_subgraph


def find_path(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, list[int]]:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    def h(a, b):
        # print(a, b)
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
            if alg == 'astar':
                return [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            raise e

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        start = time.time()
        g = layer.centroids_graph
        path = []
        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')
        if alg == 'astar':
            path = [nx.astar_path_length(g, from_center, to_center, weight='length', heuristic=h)]
        end = time.time()
        step1 = end - start
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        raise e

    start = time.time()
    cls = set()
    cls.add(to_cluster)
    for u in path:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    end = time.time()
    step2 = end - start
    try:
        start = time.time()

        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'astar':
            path = [nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h)]
        end = time.time()
        step3 = end - start
        tqdm.write(f"""
        step1: {step1}
        step2: {step2}
        step3: {step3}
        """)
        return path
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        raise e


def find_path_length(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, int, int, int]:
    alg1 = alg
    alg2 = alg
    if type(alg) is tuple:
        alg1, alg2 = alg

    from_cluster = layer.graph.nodes[from_node]['cluster']
    to_cluster = layer.graph.nodes[to_node]['cluster']

    def h(a, b):
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    if from_cluster == to_cluster:
        try:
            g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
            if alg2 == 'dijkstra':
                return nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0], 0, 0, 0
            if alg2 == 'bidirectional':
                return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0], 0, 0, 0
            if alg2 == 'astar':
                return nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h), 0, 0, 0
        except nx.NetworkXNoPath as e:
            print('No path found in one cluster')
            return -1, 0, 0, 0

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    try:
        # start = time.time()

        g = layer.centroids_graph
        path = []
        if alg1 == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg1 == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')[1]
        if alg1 == 'astar':
            path = nx.astar_path(g, from_center, to_center, weight='length', heuristic=h)
        # end = time.time()
        # step1 = end - start
    except nx.NetworkXNoPath as e:
        print('No path found in clusters')
        return -1
    # start = time.time()
    cls = set()
    cls.add(to_cluster)
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        du = layer.graph.nodes[u]
        dv = layer.graph.nodes[v]
        cu = du['cluster']
        cv = dv['cluster']
        if len(layer.cls) > 0:
            for c in layer.cls[cu][cv]:
                cls.add(c)
        cls.add(cu)
        cls.add(cv)

    def h1(a, b):
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    # end = time.time()
    # step2 = end - start
    try:
        # start = time.time()

        if alg2 == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg2 == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg2 == 'astar':
            path = nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h1)
        # end = time.time()
        # step3 = end - start
        return path, 0, 0, 0
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        return -1, 0, 0, 0


def find_path_length_h(
        layer: GraphLayer,
        from_node: int,
        to_node: int) -> float:
    from_cluster = layer.graph.nodes[from_node]['cluster']
    to_cluster = layer.graph.nodes[to_node]['cluster']
    def h(a, b):
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    g = layer.centroids_graph
    path = nx.astar_path(g, from_cluster, to_cluster, heuristic=h, weight='length')

    cls = set()
    cls.add(to_cluster)
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        cu = layer.graph.nodes[u]['cluster']
        cv = layer.graph.nodes[v]['cluster']
        cls.union(layer.cls[cu][cv])
        cls.add(cu)
        cls.add(cv)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
    return path

def find_path_length_full(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, int, int, int]:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    cls = layer.cls[from_cluster][to_cluster]

    def h1(a, b):
        da = layer.graph.nodes[a]
        db = layer.graph.nodes[b]
        return ((da['x'] - db['x']) ** 2 + (da['y'] - db['y']) ** 2) ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)
    try:

        if alg == 'dijkstra':
            path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'bidirectional':
            path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'astar':
            path = nx.astar_path_length(g, from_node, to_node, weight='length', heuristic=h1)
        return path, 0, 0, 0
    except nx.NetworkXNoPath as e:
        print(nx.is_connected(g))
        print('No path in cluster subgraph')
        return -1, 0, 0, 0
