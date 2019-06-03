import dgl
import torch


def nx_to_dgl(g):
    G = dgl.DGLGraph()
    d = {n: i for i, n in enumerate(list(g.nodes()))}
    G.add_nodes(g.number_of_nodes())
    for e in list(g.edges()):
        G.add_edge(d[e[0]], d[e[1]], {'w': torch.FloatTensor([[g[e[0]][e[1]]['weight']]])})
    return G
