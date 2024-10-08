import enum
from enum import Enum

import networkx as nx
from abc import ABC, abstractmethod
from typing import NewType, Union

from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, BisectingKMeans
import numpy as np

Community = NewType('Community', Union[list[set[int]], tuple[set[int]]])


class CommunitiesResolver(ABC):
    __slots__ = '__name'

    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    @abstractmethod
    def _resolver(self, g: nx.Graph) -> Community:
        pass

    @staticmethod
    def to_connected_communities(g: nx.Graph, communities: Community) -> Community:
        cls = []
        for i, c in enumerate(communities):
            for n in nx.connected_components(g.subgraph(c)):
                cls.append(n)
        return cls

    @staticmethod
    def write_communities(g: nx.Graph, communities: Community, field='cluster'):
        for i, ids in enumerate(communities):
            for j in ids:
                g.nodes[j][field] = i

    @staticmethod
    def extract_cluster_list_subgraph(g: nx.Graph, cluster_number: list[int] | set[int],
                                      communities: Community = None) -> nx.Graph:
        if communities:
            nodes_to_keep = [u for c in cluster_number for u in communities[c]]
        else:
            nodes_to_keep = [node for node, data in g.nodes(data=True) if data['cluster'] in cluster_number]
        return g.subgraph(nodes_to_keep)

    def find_communities(self, g: nx.Graph):
        return self.to_connected_communities(g, self._resolver(g))


class LouvainCommunitiesResolver(CommunitiesResolver):
    __slots__ = '__resolution', '__weight'

    def __init__(self, resolution: float = 1., weight='length', name: str = 'louvain'):
        super().__init__(name)
        self.resolution = resolution
        self.weight = weight

    @property
    def resolution(self) -> float:
        return self.__resolution

    @resolution.setter
    def resolution(self, value: float):
        self.__resolution = value

    @property
    def weight(self) -> str:
        return self.__weight

    @weight.setter
    def weight(self, value: str):
        self.__weight = value

    def _resolver(self, g: nx.Graph) -> Community:
        print('r:',self.resolution, self.weight)
        return nx.community.louvain_communities(g,
                                                seed=1534,
                                                weight=self.weight,
                                                resolution=self.resolution)


class KMeanCommunitiesResolver(CommunitiesResolver):

    def __init__(self, n_clusters=1, name: str = 'kmeans'):
        super().__init__(name)
        self.n_clusters = n_clusters

    def _resolver(self, g: nx.Graph) -> Community:
        kmeans = KMeans(n_clusters=self.n_clusters)
        x = np.array([[d['x'], d['y']] for u, d in g.nodes(data=True)])
        y = kmeans.fit_predict(x)
        communities = {}
        for i, u in enumerate(g.nodes):
            cls = y[i]
            if cls not in communities:
                communities[cls] = set()
            communities[cls].add(u)
        communities = [communities[cls] for cls in communities]
        return communities


class LouvainKMeansCommunitiesResolver(CommunitiesResolver):
    __slots__ = '__louvain'

    def __init__(self, resolution=1, weight='length', name: str = 'louvain_kmeans'):
        super().__init__(name)
        self.__louvain = LouvainCommunitiesResolver(resolution=resolution, weight=weight)

    def _resolver(self, g: nx.Graph) -> Community:
        communities = self.__louvain.fincommunitiesd_communities(g)

        cls2center = {}
        cls2diam = {}

        for i, cls in enumerate(communities):
            gc = self.extract_cluster_list_subgraph(g, [i], communities)
            center = nx.barycenter(gc, weight='length')[0]
            cls2center[i] = center
            l = nx.single_source_bellman_ford_path_length(gc, center, weight='length')
            cls2diam[i] = max([l[to] for to in l])

        node2cls = {}
        for i, cls in enumerate(communities):
            center = cls2center[i]
            l = nx.single_source_dijkstra_path_length(g, center, cutoff=cls2diam[i], weight='length')
            for u in l:
                if u not in node2cls:
                    node2cls[u] = {}
                node2cls[u][i] = l[u]
        node2cls = {u: min(node2cls[u].items(), key=lambda x: x[1])[0] for u in node2cls}
        communities = {}
        for node in node2cls:
            if node2cls[node] not in communities:
                communities[node2cls[node]] = set()
            communities[node2cls[node]].add(node)
        communities = [communities[cls] for cls in communities]
        return communities


class GreedyModularityCommunitiesResolver(LouvainCommunitiesResolver):
    def __init__(self, resolution: float = 1., weight: str = 'length', name: str = 'greedy_modularity'):
        super().__init__(resolution=resolution, weight=weight, name=name)

    def _resolver(self, g: nx.Graph) -> Community:
        return nx.community.greedy_modularity_communities(
            g,
            weight=self.weight,
            resolution=self.resolution
        )


class DBScanCommunitiesResolver(CommunitiesResolver):

    def __init__(self, eps: float = 0.2, min_samples=2, name='dbscan'):
        super().__init__(name)
        self.eps = eps
        self.min_samples = min_samples

    def _resolver(self, g: nx.Graph) -> Community:
        scan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        x = np.array([[d['x'], d['y']] for u, d in g.nodes(data=True)])
        y = scan.fit_predict(x)
        communities = {}
        for i, u in enumerate(g.nodes):
            cls = y[i]
            if cls not in communities:
                communities[cls] = set()
            communities[cls].add(u)
        communities = [communities[cls] for cls in communities]
        return communities


class HDBScanCommunitiesResolver(CommunitiesResolver):

    def __init__(self, min_cluster_size: int = 10, name='hdbscan'):
        super().__init__(name)
        self.min_cluster_size = min_cluster_size

    def _resolver(self, g: nx.Graph) -> Community:
        scan = HDBSCAN(min_cluster_size=self.min_cluster_size)
        x = np.array([[d['x'], d['y']] for u, d in g.nodes(data=True)])
        y = scan.fit_predict(x)
        communities = {}
        for i, u in enumerate(g.nodes):
            cls = y[i]
            if cls not in communities:
                communities[cls] = set()
            communities[cls].add(u)
        communities = [communities[cls] for cls in communities]
        return communities


class BisectingKMeansCommunitiesResolver(KMeanCommunitiesResolver):
    def __init__(self, n_clusters=1, name: str = 'bisecting_kmeans'):
        super().__init__(n_clusters=n_clusters, name=name)

    def _resolver(self, g: nx.Graph) -> Community:
        try:
            kmeans = BisectingKMeans(n_clusters=self.n_clusters, random_state=0)
            x = np.array([[d['x'], d['y']] for u, d in g.nodes(data=True)])
            y = kmeans.fit_predict(x)
        except:
            return [{u} for u in g.nodes]

        communities = {}
        for i, u in enumerate(g.nodes):
            cls = y[i]
            if cls not in communities:
                communities[cls] = set()
            communities[cls].add(u)
        communities = [communities[cls] for cls in communities]
        return communities


@enum.unique
class Method(Enum):
    LOUVAIN = LouvainCommunitiesResolver,
    K_MEAN = KMeanCommunitiesResolver,
    BISECTING_K_MEAN = BisectingKMeansCommunitiesResolver,
    LOUVAIN_K_MEANS = LouvainKMeansCommunitiesResolver,
    GREEDY_MODULARITY = GreedyModularityCommunitiesResolver,
    DBSCAN = DBScanCommunitiesResolver,
    HDBSCAN = HDBScanCommunitiesResolver,


def get_method(method: Method, **params) -> CommunitiesResolver:
    return method.value[0](**params)
