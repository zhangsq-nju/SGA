from networkx.readwrite import json_graph


def node_link_graph_compat(data):
    if "links" in data:
        return json_graph.node_link_graph(data, edges="links")
    if "edges" in data:
        return json_graph.node_link_graph(data, edges="edges")
    return json_graph.node_link_graph(data)
