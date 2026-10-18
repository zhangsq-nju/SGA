import json

import numpy as np
import networkx as nx
from dtaidistance import dtw
from networkx.readwrite import json_graph
from tqdm import tqdm

class GraphConstructor:
    def __init__(self):
        self.merge_policy = "same-time"
        self.sim_tool = "dtw"
        self.slice_length = 4
        self.threshold_coefficient = 0.25
        self.dtw_window = None
        self.dtw_max_step = None
        self.scale_normalization = "none"
        self.scale_eps = 1e-4
        self.global_sigma = 1.0

    def __call__(self, responses_data, output_dir, graph_cfg=None):
        if graph_cfg is None:
            raise ValueError("graph_cfg is required for SGA graph construction")
        if "merge_way" in graph_cfg or "sim_way" in graph_cfg:
            raise ValueError("Deprecated graph_cfg keys 'merge_way/sim_way' are not supported. Use 'merge_policy/sim_tool'.")

        self.merge_policy = graph_cfg.merge_policy
        self.sim_tool = graph_cfg.sim_tool
        if self.merge_policy != "same-time":
            raise ValueError("SGA uses same-time graph merging")
        if self.sim_tool != "dtw":
            raise ValueError("sim_tool currently supports only 'dtw'")
        self.slice_length = graph_cfg.slice_length
        self.threshold_coefficient = graph_cfg.threshold_coefficient
        self.dtw_window = graph_cfg.get("dtw_window", None)
        self.dtw_max_step = graph_cfg.get("dtw_max_step", None)
        self.scale_normalization = graph_cfg.get("scale_normalization", "none")
        self.scale_eps = float(graph_cfg.get("scale_eps", 1e-4))
        if self.scale_normalization not in {"none", "seasonal_error"}:
            raise ValueError("scale_normalization must be one of {'none', 'seasonal_error'}")



        g_data = []
        output_path = (
            f"{output_dir}/{self.merge_policy}_{self.sim_tool}/"
            f"g_data_{self.slice_length}_{self.threshold_coefficient}_{self.dtw_window}_{self.scale_normalization}"
            ".jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for item in tqdm(responses_data, desc="graph_construct"):
                series_id = item["id"]
                responses = item["responses"]
                G, terminal_ids = self._construct(responses, seasonal_error=item.get("seasonal_error", None))
                g_item = {
                    "id": series_id,
                    "G": json_graph.node_link_data(G),
                    "terminal_node_ids": terminal_ids,
                    "num_series": len(responses),
                }
                g_data.append(g_item)
                f.write(json.dumps(g_item, ensure_ascii=False) + "\n")

        return g_data

    def _get_similarity_scale(self, seasonal_error):
        if self.scale_normalization == "none":
            return 1.0
        if seasonal_error is None:
            raise ValueError("scale_normalization='seasonal_error' requires item['seasonal_error']")
        scale = float(seasonal_error)
        if not np.isfinite(scale):
            raise ValueError(f"Invalid seasonal_error: {seasonal_error}")
        return max(abs(scale), self.scale_eps)

    def _construct(self, responses, seasonal_error=None):
        G = nx.DiGraph()
        G.add_node(0, num_nodes=1)
        node_id = 1
        similarity_scale = self._get_similarity_scale(seasonal_error)
        node_sim_slices = {}
        for chain_id, response in enumerate(responses):
            slices = self._slice_trajectory(response, similarity_scale=similarity_scale)
            prev_id = 0
            for slice_index, slice in enumerate(slices):
                nodelist = [
                    node for node in G.nodes
                    if node != 0 and G.nodes[node].get("slice_index", -1) == slice_index
                ]
                merge_node = self._find_merge_node(node_sim_slices, nodelist, slice["sim_values"])
                if merge_node is not None:
                    if prev_id != merge_node:
                        G.add_edge(prev_id, merge_node, chain=chain_id)
                        G.nodes[merge_node]["prob_info_all"].append(slice["prob info"])
                        G.nodes[merge_node]["num_nodes"] += 1
                    prev_id = merge_node
                else:
                    G.add_node(
                        node_id,
                        slice=slice["values"],
                        slice_index=slice_index,
                        num_nodes=1,
                        prob_info_all=[slice["prob info"]],
                    )
                    G.add_edge(prev_id, node_id, chain=chain_id)
                    node_sim_slices[node_id] = slice["sim_values"]
                    prev_id = node_id
                    node_id += 1
        return G, self._get_terminal_nodes(G)

    def _slice_trajectory(self, response, similarity_scale=1.0):
        series = [float(v) for v in response["predicted series"]]
        if "prob_info" not in response:
            raise ValueError("Graph construction now requires precomputed response['prob_info']")

        slices = []
        if not self.slice_length or self.slice_length <= 0:
            self.slice_length = len(series)
        for start in range(0, len(series), self.slice_length):
            end = min(start + self.slice_length, len(series))
            slice_values = series[start:end]
            slice_sim_values = [v / similarity_scale for v in slice_values]
            slices.append(
                {
                    "values": slice_values,
                    "sim_values": slice_sim_values,
                    "prob info": self._slice_precomputed_prob_info(response, start, end),
                }
            )

        return slices

    def _slice_precomputed_prob_info(self, response, start, end):
        values = response["prob_info"]["neg entropy"]
        if values and len(values) != len(response["predicted series"]):
            raise ValueError("Length mismatch between predicted series and entropy")
        return {"neg entropy": values[start:end]}

    def _find_merge_node(self, node_sim_slices, nodelist, new_slice):
        for u in nodelist:
            slice = node_sim_slices.get(u, None)
            if slice is None:
                continue
            if self._are_slices_similar(slice, new_slice):
                return u
        return None

    def _are_slices_similar(self, seq1, seq2):
        if not seq1 or not seq2:
            return False
        distance = dtw.distance(
            np.asarray(seq1, dtype=np.float64),
            np.asarray(seq2, dtype=np.float64),
            window=self.dtw_window,
            max_step=self.dtw_max_step,
            use_c=True,
        )
        normalized_distance = distance / self.global_sigma / max(len(seq1), len(seq2))
        return bool(normalized_distance <= self.threshold_coefficient)

    def _get_terminal_nodes(self, G):
        sinks = [u for u in G.nodes if u != 0 and G.out_degree(u) == 0]
        return sinks if sinks else [0]
