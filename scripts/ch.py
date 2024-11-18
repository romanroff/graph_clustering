import networkx as nx
from tqdm import trange
from heapq import heappop, heappush
from itertools import count


class ChDiGraph:
    def __init__(self, dg: nx.DiGraph, edges_to_node: dict[tuple[int, int], int]):
        self.dg: nx.DiGraph = dg
        self.edges_to_node: dict[tuple[int, int], int] = edges_to_node


def _add_edges(graph: nx.Graph,
               c_graph: nx.Graph,
               node: int,
               edges_to_nodes: dict[tuple[int, int], int]):
    U = c_graph[node]
    W = c_graph[node]

    c_graph.remove_node(node)
    if len(U) <= 1:
        return

    P = {(u, w): U[u]['length'] + W[w]['length'] for u in U for w in W if u != w}
    P_MAX = max(P.values()) + 1
    NEW_EDGES = {}

    for u in U:
        paths = nx.single_source_dijkstra_path_length(c_graph, u, weight='length', cutoff=P_MAX)
        for w in W:
            if w == u:
                continue
            if w not in paths or paths[w] > P[u, w]:
                NEW_EDGES[u, w] = P[u, w]
    for (u, w), l in NEW_EDGES.items():
        edges_to_nodes[u, w] = node
        edges_to_nodes[w, u] = node
        graph.add_edge(u, w, length=l)
        c_graph.add_edge(u, w, length=l)


def build_ch_graph(graph: nx.Graph) -> ChDiGraph:
    edges_to_nodes: dict[tuple[int, int], int] = {}
    gg = graph.copy()
    cg = graph.copy()
    for i in trange(len(graph.nodes)):
        nodes = [(u, d) for u, d in nx.degree(cg)]
        u = min(nodes, key=lambda x: x[1])[0]
        gg.nodes()[u]['i'] = i
        _add_edges(gg, cg, u, edges_to_nodes)
    ch_graph = nx.DiGraph()
    for u, du in gg.nodes(data=True):
        ch_graph.add_node(u, **du)
        for v, d in gg[u].items():
            if gg.nodes()[v]['i'] > gg.nodes()[u]['i']:
                ch_graph.add_edge(u, v, length=d['length'])
    del gg, cg
    return ChDiGraph(ch_graph, edges_to_nodes)


def get_path(u1, u2, edges_to_nodes):
    if (u1, u2) in edges_to_nodes:
        u = edges_to_nodes[u1, u2]
        return get_path(u1, u, edges_to_nodes) + get_path(u, u2, edges_to_nodes)
    return [u1]


def ch_pfa(
        graph: ChDiGraph,
        start: int,
        end: int,
        cms: set[int] | None = None
) -> tuple[float, list[int]]:
    if start == end:
        return 0, [start]
    adjacency = graph.dg._adj
    nodes = graph.dg.nodes()
    edges_to_nodes = graph.edges_to_node
    push = heappush
    pop = heappop
    dist = (set(), set())
    fringe = ([], [])
    c = count()

    push(fringe[0], (0, next(c), 0, start))
    push(fringe[1], (0, next(c), 0, end))

    heads = [0, 0]
    seens = ({start: (0, None, 0)}, {end: (0, None, 0)})
    union_node = None
    union_dst = float('inf')
    dir = 1
    while fringe[0] or fringe[1]:
        if fringe[0] and fringe[1]:
            dir = 1 - dir
        elif fringe[0]:
            dir = 0
        else:
            dir = 1

        (d, _, n, v) = pop(fringe[dir])

        heads[dir] = d

        if v in dist[dir]:
            continue

        dist[dir].add(v)

        for u, l in adjacency[v].items():
            if cms and nodes[u]['cluster'] not in cms:
                continue
            vu_dist = d + l['length']
            if u not in dist[dir] and (u not in seens[dir] or seens[dir][u][0] > vu_dist):
                seens[dir][u] = (vu_dist, v, n + 1)
                push(fringe[dir], (vu_dist, next(c), n + 1, u))
                if u in seens[1 - dir]:
                    tpl = seens[1 - dir][u]
                    dd = tpl[0] + vu_dist
                    if dd < union_dst:
                        union_dst = dd
                        union_node = u
        if min(heads) > union_dst:
            break
    # todo optimize для мск 6.5 и 6.7 с и без поиска пути
    path = []
    e = union_node
    while seens[0][e][1] is not None:
        e1 = seens[0][e][1]
        p = get_path(e1, e, edges_to_nodes)
        path = p + path
        e = e1
    e = union_node
    while seens[1][e][1] is not None:
        e1 = seens[1][e][1]
        p = get_path(e, e1, edges_to_nodes)
        path += p
        e = e1
    path += [end]
    return union_dst, path
