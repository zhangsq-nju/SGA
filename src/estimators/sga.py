import networkx as nx

from .estimator import Estimator
from src.utils.graph_io import node_link_graph_compat

class SGA(Estimator):
    def __init__(self, aggregation="all-sum", **kwargs):
        super().__init__(**kwargs)
        if aggregation != "all-sum":
            raise ValueError("SGA uses all-sum aggregation")
        self.aggregation = aggregation
        self.alpha = kwargs.get("alpha", 0.1)

    def __str__(self):
        return "SGA"

    def __call__(self, g_data):
        u_dict = {}
        for item in g_data:
            series_id = item["id"]
            G = node_link_graph_compat(item["G"])
            u = self._graph_complexity(G)
            u_dict[series_id] = u

        return u_dict

    def _graph_complexity(self, G):
        self._alpha_centrality(G)
        return sum(G.nodes[u]['alpha_centrality'] for u in G.nodes)


    def _alpha_centrality(self, G):
        for u in nx.topological_sort(G):
            s = self._get_weight(G.nodes[u])
            for v in G.predecessors(u):
                s += self.alpha * G.nodes[v]['alpha_centrality']
            G.nodes[u]['alpha_centrality'] = s
