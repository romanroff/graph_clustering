from heapq import heappop, heappush
from itertools import count
import networkx as nx


def dijkstra_pfa(graph: nx.Graph,
                 start: int,
                 end: int,
                 cms: set[int] | None = None) -> \
        tuple[float, list[int]]:
    if start == end:
        return 0, [start]
    adjacency = graph._adj
    nodes = graph.nodes()
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    pred = {}
    fringe = []
    push(fringe, (0.0, next(c), 0, start, None))
    while fringe:
        (d, _, n, v, p) = pop(fringe)
        if v in dist:
            continue
        dist[v] = (d, n)
        pred[v] = p
        if v == end:
            break
        for u, e in adjacency[v].items():
            if cms and nodes[u]['cluster'] not in cms:
                continue
            vu_dist = d + e['length']
            if u not in dist:
                push(fringe, (vu_dist, next(c), n + 1, u, v))
    d, n = dist[end]
    n += 1
    path = [None] * n
    i = n - 1
    e = end
    while i >= 0:
        path[i] = e
        i -= 1
        e = pred[e]
    return d, path


def bi_dijkstra_pfa(graph: nx.Graph,
                    start: int,
                    end: int,
                    cms: set[int] | None = None
                    ) -> tuple[float, list[int]]:
    if start == end:
        return 0, [start]
    push = heappush
    pop = heappop
    dist = ({start: (0, 0, None)}, {end: (0, 0, None)})
    fringe = ([], [])
    c = count()

    adjacency = graph._adj
    nodes = graph.nodes()

    push(fringe[0], (0, next(c), 0, start))
    push(fringe[1], (0, next(c), 0, end))

    union_node = None
    union_dst = float('inf')
    while fringe[0] and fringe[1]:
        (d1, _, n1, v1) = pop(fringe[0])
        (d2, _, n2, v2) = pop(fringe[1])
        for u, e in adjacency[v1].items():
            if cms and nodes[u]['cluster'] not in cms:
                continue

            vu_dist = d1 + e['length']
            if u not in dist[0] or dist[0][u][0] > vu_dist:
                dist[0][u] = (vu_dist, n1 + 1, v1)
                push(fringe[0], (vu_dist, next(c), n1 + 1, u))
            if u in dist[1]:
                dd = dist[1][u][0] + dist[0][u][0]
                if dd < union_dst:
                    union_dst = dd
                    union_node = u
        for u, e in adjacency[v2].items():
            if cms and nodes[u]['cluster'] not in cms:
                continue

            vu_dist = d2 + e['length']
            if u not in dist[1] or dist[1][u][0] > vu_dist:
                dist[1][u] = (vu_dist, n2 + 1, v2)
                push(fringe[1], (vu_dist, next(c), n2 + 1, u))
            if u in dist[0]:
                dd = dist[0][u][0] + dist[1][u][0]
                if dd < union_dst:
                    union_dst = dd
                    union_node = u
        if d1 + d2 > union_dst:
            break

    d1, n1, _ = dist[0][union_node]
    d2, n2, _ = dist[1][union_node]
    path = [0] * (n1 + n2 + 1)
    e = union_node
    i = n1
    while dist[0][e][2] is not None:
        path[i] = e
        i -= 1
        e = dist[0][e][2]
    path[0] = e

    e = union_node
    i = n1
    while dist[1][e][2] is not None:
        path[i] = e
        i += 1
        e = dist[1][e][2]
    path[-1] = e
    return union_dst, path
