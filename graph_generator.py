import itertools
import math
import random
import time

import networkx as nx
import numpy as np
import osmnx as ox
from tqdm import tqdm

import communities_resolver
from common import GraphLayer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN, MeanShift


def get_dist(du, dv) -> float:
    d = (du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2
    d = d ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000
    return d


def extract_cluster_subgraph(graph: nx.Graph, cluster_number: int) -> nx.Graph:
    nodes_to_keep = [node for node, data in graph.nodes(data=True) if data['cluster'] == cluster_number]
    return graph.subgraph(nodes_to_keep)


def extract_cluster_list_subgraph(graph: nx.Graph, cluster_number: list[int] | set[int], communities=None) -> nx.Graph:
    if communities:
        nodes_to_keep = [u for c in cluster_number for u in communities[c]]
    else:
        nodes_to_keep = [node for node, data in graph.nodes(data=True) if data['cluster'] in cluster_number]
    return graph.subgraph(nodes_to_keep)


def get_graph(city_id: str = 'R2555133') -> nx.Graph:
    gdf = ox.geocode_to_gdf(city_id, by_osmid=True)
    polygon_boundary = gdf.unary_union
    graph = ox.graph_from_polygon(polygon_boundary,
                                  network_type='drive',
                                  simplify=True)
    G = nx.Graph(graph)
    H = nx.Graph()
    # Добавляем рёбра в новый граф, копируя только веса
    for u, d in G.nodes(data=True):
        H.add_node(u, x=d['x'], y=d['y'])
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, length=d['length'])
    del city_id, gdf, polygon_boundary, graph, G
    return H


def resolve_communities(H: nx.Graph, method: str = 'louvain', **params) -> list[set[int]]:
    if method == 'louvain':
        communities = nx.community.louvain_communities(H,
                                                       seed=1534,
                                                       weight='length',
                                                       resolution=params['r'])
    elif method == 'louvain_knn':
        communities = communities_resolver.louvain_kmeans(H,r = params['r'])
    elif method == 'kmeans':
        communities = communities_resolver.knn(H, **params)
    elif method == 'dbscan':
        communities = communities_resolver.dbscan(H, **params)
    else:
        communities = nx.community.greedy_modularity_communities(H,
                                                                 weight='length',
                                                                 resolution=params['r'])
    cls = []

    for i, c in enumerate(communities):
        for n in nx.connected_components(H.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            H.nodes[j]['cluster'] = i

    # cls2hubs = get_cluster_to_bridge_points(H)
    #
    # cls_new = []
    # added = set()
    #
    # for i, ids in enumerate(cls):
    #     c = ids.copy()
    #     for node in cls2hubs[i]:
    #         c.remove(node)
    #         if node not in added:
    #             cls_new.append({node})
    #         added.add(node)
    #     if len(c) > 0:
    #         cls_new.append(c)
    #
    # for i, ids in enumerate(cls_new):
    #     for j in ids:
    #         H.nodes[j]['cluster'] = i
    #
    # cls = []
    #
    # for i, c in enumerate(communities):
    #     for n in nx.connected_components(H.subgraph(c)):
    #         cls.append(n)
    # for i, ids in enumerate(cls):
    #     for j in ids:
    #         H.nodes[j]['cluster'] = i

    return cls


def generate_communities_subgraph(H: nx.Graph, communities: list[set[int]]) -> list[nx.Graph]:
    return [extract_cluster_subgraph(H, i) for i, c in enumerate(communities)]


def get_cluster_to_neighboring_clusters(H: nx.Graph) -> dict[int, set[int]]:
    cls_to_neighboring_cls = {}
    for u in H.nodes():
        du = H.nodes[u]
        for v in H[u]:
            dv = H.nodes[v]
            if dv['cluster'] == du['cluster']:
                continue
            c1 = dv['cluster']
            c2 = du['cluster']
            if not (c1 in cls_to_neighboring_cls):
                cls_to_neighboring_cls[c1] = set()
            if not (c2 in cls_to_neighboring_cls):
                cls_to_neighboring_cls[c2] = set()
            cls_to_neighboring_cls[c1].add(c2)
            cls_to_neighboring_cls[c2].add(c1)
    return cls_to_neighboring_cls


def get_cluster_to_bridge_points(H: nx.Graph) -> dict[int, set[int]]:
    cls_to_bridge_points = {}
    for u in H.nodes():
        from_node = H.nodes[u]
        for v in H[u]:
            to_node = H.nodes[v]
            c1 = from_node['cluster']
            c2 = to_node['cluster']

            if c1 == c2:
                continue

            if not (c1 in cls_to_bridge_points):
                cls_to_bridge_points[c1] = set()
            if not (c2 in cls_to_bridge_points):
                cls_to_bridge_points[c2] = set()
            cls_to_bridge_points[c1].add(u)
            cls_to_bridge_points[c2].add(v)
    return cls_to_bridge_points


def get_cluster_to_centers(X: nx.Graph) -> dict[int, int]:
    cls_to_center = {d['cluster']: u for u, d in X.nodes(data=True)}
    return cls_to_center


def get_path_len(d: dict, points, p=1.0):
    res_len = 0
    for u in points:
        if u in d and d[u] > 0:
            res_len += d[u] ** p
    if res_len > 0:
        return res_len ** (1 / p)
    return 0


def build_center_graph(
        graph: nx.Graph,
        communities: list[set[int]],
        cluster_to_bridge_points: dict[int, set[int]],
        cluster_to_neighboring_cluster: dict[int, set[int]],
        p: float = 1.0,
        use_all_point: bool = True,
        has_coordinates: bool = True,
        optimal_path: str = '',
        paths=None
) -> tuple[nx.Graph, dict[int:dict[int, set[int]]]]:
    """
        строим граф центройд по комьюнити для графа G
    """
    centers = {}
    X = nx.Graph()
    for cls, _ in enumerate(communities):
        gc = extract_cluster_list_subgraph(graph, [cls], communities)
        # if has_coordinates:
        #     _p: dict[int, dict[int, float]] = {u: {v: get_dist(du, dv) for v, dv in gc.nodes(data=True)} for u, du in
        #                                        gc.nodes(data=True)}
        # else:
        #     _p: dict[int, dict[int, float]] = dict(nx.all_pairs_bellman_ford_path_length(gc, weight='length'))
        # if use_all_point:
        #     dist = {u: get_path_len(_p[u], communities[cls], p) for u in _p}
        # else:
        #     dist = {u: get_path_len(_p[u], cluster_to_bridge_points[cls], p) for u in _p}
        # min_path = None
        # min_node = 0
        # for u in dist:
        #     d = dist[u]
        #     if min_path is None or d < min_path:
        #         min_path = d
        #         min_node = u
        min_node = nx.barycenter(gc, weight='length')[0]
        du = graph.nodes(data=True)[min_node]
        X.add_node(min_node, **du)
        centers[cls] = min_node
    if len(X.nodes) == 1:
        return X, {}
    data: dict[int:dict[int, set[int]]] = {}
    if optimal_path == 'second':
        for u, du in tqdm(X.nodes(data=True), total=len(X.nodes), position=2):
            for v, dv in X.nodes(data=True):
                cu = du['cluster']
                cv = dv['cluster']
                if cu not in data:
                    data[cu] = {}
                if cv not in data[cu]:
                    data[cu][cv] = set()

                if cu == cv:
                    data[cu][cv].add(cu)
                    continue
                if paths is not None:
                    path = paths[u][v]
                else:
                    path = nx.single_source_dijkstra(
                        graph,
                        u,
                        v,
                        weight='length'
                    )[1]
                cls = set([graph.nodes[path_node]['cluster'] for path_node in path])
                for path_cluster in cls:
                    data[cu][cv].add(path_cluster)
        return X, data
    else:
        for u, du in X.nodes(data=True):
            for cls_to in cluster_to_neighboring_cluster[du['cluster']]:
                v = centers[cls_to]
                dv = X.nodes[centers[cls_to]]
                cu = du['cluster']
                cv = dv['cluster']

                if optimal_path == 'first':
                    path = nx.single_source_dijkstra(
                        graph,
                        u,
                        v,
                        weight='length'
                    )
                    if cu not in data:
                        data[cu] = {}
                    if cv not in data[cu]:
                        data[cu][cv] = set()

                    cls = set([graph.nodes[path_node]['cluster'] for path_node in path[1]])
                    for path_cluster in cls:
                        data[cu][cv].add(path_cluster)
                    path = path[0]
                elif has_coordinates:
                    path = np.sqrt((du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2)
                else:
                    path = nx.single_source_dijkstra(
                        extract_cluster_list_subgraph(graph, [du['cluster'], dv['cluster']], communities),
                        u,
                        centers[cls_to],
                        weight='length'
                    )[0]
                X.add_edge(u, centers[cls_to], length=path)
        l = [len(u[v]) for u in data.values() for v in u]
        # print(np.mean(l), np.std(l))
        return X, data


def get_node(H: nx.Graph, cls_to_center: dict, X: nx.Graph):
    node_from = random.choice(list(H.nodes()))
    node_to = random.choice(list(H.nodes()))
    path_len = nx.single_source_dijkstra(H, node_from, node_to, weight='length')
    c = set()
    for u in path_len[1]:
        c.add(H.nodes[u]['cluster'])
    while len(c) < 5:
        node_from = random.choice(list(H.nodes()))
        node_to = random.choice(list(H.nodes()))
        path_len = nx.single_source_dijkstra(H, node_from, node_to, weight='length')
        c.clear()
        for u in path_len[1]:
            c.add(H.nodes[u]['cluster'])
    return node_from, node_to


def get_node_for_initial_graph_v2(H: nx.Graph):
    nodes = list(H.nodes())
    f, t = random.choice(nodes), random.choice(nodes)
    while f == t:
        f, t = random.choice(nodes), random.choice(nodes)
    return f, t


def get_node_for_initial_graph(H: nx.Graph):
    R = nx.radius(H, weight='length')
    path_len = R / 2
    nodes = list(H.nodes())
    while True:
        for i in range(100):
            node_from = random.choice(nodes)
            node_to = random.choice(nodes)
            current_len = nx.single_source_dijkstra(H, node_from, node_to)[0]
            if current_len >= path_len:
                return node_from, node_to
        path_len = path_len // 3 * 2


def generate_layer(H: nx.Graph, resolution: float, p: float = 1, use_all_point: bool = True, communities=None,
                   has_coordinates: bool = False, paths=None) -> GraphLayer:
    start = time.time()
    if communities is None:
        communities = resolve_communities(H, r=resolution)
    build_communities = time.time() - start
    start = time.time()
    cluster_to_neighboring_clusters = get_cluster_to_neighboring_clusters(H)
    cluster_to_bridge_points = get_cluster_to_bridge_points(H)
    build_additional = time.time() - start
    start = time.time()

    centroids_graph, data = build_center_graph(
        graph=H,
        communities=communities,
        cluster_to_bridge_points=cluster_to_bridge_points,
        cluster_to_neighboring_cluster=cluster_to_neighboring_clusters,
        p=p,
        use_all_point=use_all_point,
        has_coordinates=has_coordinates,
        paths=paths
    )
    build_centroid_graph = time.time() - start

    cluster_to_centers = get_cluster_to_centers(centroids_graph)

    layer: GraphLayer = GraphLayer(
        H,
        communities,
        cluster_to_neighboring_clusters,
        cluster_to_bridge_points,
        cluster_to_centers,
        centroids_graph
    )
    layer.cls = data
    return layer, build_communities, build_additional, build_centroid_graph
