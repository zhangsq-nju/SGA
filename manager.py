import os
import json
import re
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.estimators import *
from src.ue_metrics import *
from src.utils.log import setup_log
from src.utils.load_jsonl import load_jsonl
from src.utils.write_tsv import write_result_to_tsv
from src.utils.match_values import match_values
from src.utils.series_metadata import load_series_metadata, merge_series_metadata, write_series_metadata
from src.utils.ts_eval import calculate_series_error


def get_inference_output_name(cfg):
    if HydraConfig.initialized():
        inference_choice = HydraConfig.get().runtime.choices.get("inference", None)
        print(f"Inference choice from Hydra: {inference_choice}")
        if inference_choice:
            return inference_choice

    fallback = cfg.inference.get("dataset_name", "inference")
    if cfg.inference.get("subset", None):
        return f"{fallback}-{cfg.inference.subset}"
    if cfg.inference.get("target_col", "target") != "target":
        return f"{fallback}-{cfg.inference.target_col}"
    return fallback


def model_requires_nan_imputation(model_name):
    from src.utils.model_init import infer_backend_from_model_name

    return infer_backend_from_model_name(model_name) in {"aurora", "sundial", "timer_s1"}


def inference(cfg, log, output_dir):
    do_inference_once = cfg.must_inference_once or not os.path.isfile(f"{output_dir}/response_data.jsonl")
    do_inference_multiple = cfg.must_inference_multiple or not os.path.isfile(f"{output_dir}/responses_data.jsonl")

    metadata = load_series_metadata(output_dir)
    if do_inference_once or do_inference_multiple or not metadata:
        from src.utils.load_data import load_data, impute_contexts_last_value
        data = load_data(cfg.inference)
        metadata = {
            item["id"]: item
            for item in write_series_metadata(output_dir, data)
        }
        metadata |= {str(k): v for k, v in metadata.items()}
        if model_requires_nan_imputation(cfg.model_name):
            data = impute_contexts_last_value(data)

    if do_inference_once or do_inference_multiple:
        from src.utils.model_init import load_model
        model = load_model(cfg)
    # raise ValueError("This code is deprecated.")

    if do_inference_once:
        from src.stat_calculators.inference_once import model_inference_once

        response_data = model_inference_once(cfg.inference, model, data, log, output_dir)
    else:
        response_data = load_jsonl(f"{output_dir}/response_data.jsonl")

    if do_inference_multiple:
        from src.stat_calculators.inference_multiple import model_inference_multiple

        responses_data = model_inference_multiple(cfg.inference, model, data, log, output_dir)
    else:
        responses_data = load_jsonl(f"{output_dir}/responses_data.jsonl")

    return merge_series_metadata(response_data, metadata), merge_series_metadata(responses_data, metadata)


def g_construct(cfg, responses_data, output_dir):
    g_output_dir = f"{output_dir}/{cfg.graph_cfg.merge_policy}_{cfg.graph_cfg.sim_tool}"
    # graph_alpha = float(cfg.graph_cfg.get("alpha", 0.0))
    # graph_drop_threshold = float(cfg.graph_cfg.get("drop_threshold", 0.1))
    g_output_path = (
        f"{g_output_dir}/g_data_"
        f"{cfg.graph_cfg.slice_length}_{cfg.graph_cfg.threshold_coefficient}_{cfg.graph_cfg.get('dtw_window', None)}_"
        f"{cfg.graph_cfg.get('scale_normalization', 'none')}"
        # f"{graph_alpha}_{graph_drop_threshold}"
        ".jsonl"
    )
    if cfg.must_construct_graph or not os.path.exists(g_output_path):
        from src.stat_calculators.graph_construct import GraphConstructor

        graph_construct = GraphConstructor()
        os.makedirs(g_output_dir, exist_ok=True)
        g_data = graph_construct(responses_data, output_dir, cfg.graph_cfg)
    else:
        g_data = load_jsonl(g_output_path)

    return g_data


def get_estimators(cfg):
    Est_dict = Estimator.get_subclass_dict()
    est_list = []
    for ecfg in cfg.estimators:
        if ecfg.name not in Est_dict:
            raise ValueError(f"Estimator '{ecfg.name}' not found. Make sure it is imported.")
        Est = Est_dict[ecfg.name]
        est = Est(**ecfg.cfg)
        est_list.append(est)
    return est_list


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")


def dump_u_dict_if_needed(cfg, output_dir, est, u_dict):
    if not cfg.get("dump_u_dict", False):
        return
    path = os.path.join(
        output_dir,
        f"u_dict_{_safe_filename(cfg.result_name)}_{_safe_filename(str(est))}.json",
    )
    serializable = {
        str(k): float(v)
        for k, v in sorted(u_dict.items(), key=lambda item: str(item[0]))
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, sort_keys=True)


def manager(cfg):
    from src.utils.randomness import resolve_seed, set_global_seed
    set_global_seed(resolve_seed(cfg.inference))
    with open_dict(cfg.inference):
        cfg.inference.batch_size = int(cfg.get("batch_size", 32))

    inference_name = get_inference_output_name(cfg)
    output_dir = f"./output/{cfg.exp_name}/{inference_name}/{cfg.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # if cfg.need_main and os.path.isfile(f"{output_dir}/results_{cfg.result_name}.tsv"):
    #     os.remove(f"{output_dir}/results_{cfg.result_name}.tsv")

    log = setup_log(output_dir)

    response_data, responses_data = inference(cfg, log, output_dir)
    # raise ValueError("This code is deprecated.")
    error_dict = calculate_series_error(response_data, cfg.error_metric, seasonality=cfg.get("mase_seasonality", None))
    mean_error = sum(error_dict.values()) / len(error_dict)
    log.debug(f"{cfg.error_metric.upper()}: {mean_error:.6f}")

    g_data = g_construct(cfg, responses_data, output_dir)

    metrics = {
        # "Pearson": PearsonCorrelation(),
        "Spearman": SpearmanRankCorrelation(),
        "AURC": AURC(
            error_threshold=cfg.aurc_error_threshold,
        ),
        "EAURC": EAURC(
            error_threshold=cfg.aurc_error_threshold,
        ),
        "NEAURC": NEAURC(
            error_threshold=cfg.aurc_error_threshold,
        ),
    }

    g_name = (
        f"{cfg.graph_cfg.merge_policy}_{cfg.graph_cfg.sim_tool}"
        f"_{cfg.graph_cfg.slice_length}_{cfg.graph_cfg.threshold_coefficient}_{cfg.graph_cfg.get('dtw_window', None)}"
        f"_{cfg.graph_cfg.get('scale_normalization', 'none')}"
        # f"_{float(cfg.graph_cfg.get('alpha', 0.0))}_{float(cfg.graph_cfg.get('drop_threshold', 0.1))}"
    )
    est_list = get_estimators(cfg)
    for est in est_list:
        u_dict = est(g_data)
        dump_u_dict_if_needed(cfg, output_dir, est, u_dict)

        if cfg.need_main:
            mean_u = sum(u_dict.values()) / len(u_dict)
            us, errors = match_values(u_dict, error_dict)

            scores = {}
            for metric_name, metric in metrics.items():
                scores[metric_name] = f"{metric(us, errors):.6f}"
                if cfg.need_std and metric_name == "NEAURC":
                    scores[f"{metric_name}_STD"] = f"{bootstrap_std(np.array(us), np.array(errors), metric, cfg=cfg):.6f}"

            log.debug(
                f"graph_name: {g_name}, estimator: {str(est)}, token_level: {est.token_level}, step_aggregation: {est.step_aggregation}"
            )
            for metric_name, score in scores.items():
                log.debug(f"{metric_name}: {score}")
            log.debug(f"Mean_Uncertainty: {mean_u:.6f}")

            write_result_to_tsv(
                f"{output_dir}/results_{cfg.result_name}.tsv",
                {
                    "graph_name": g_name,
                    "estimator": str(est),
                    "error_metric": cfg.error_metric,
                }
                | scores
                | {
                    "Mean_Uncertainty": f"{mean_u:.6f}",
                    cfg.error_metric.upper(): f"{sum(errors) / len(errors):.6f}",
                    "est_params_tag": (
                        f"alpha={getattr(est, 'alpha', '-')};"
                        f"beta={getattr(est, 'beta', '-')};"
                        f"token_level={getattr(est, 'token_level', '-')};"
                        f"step_aggregation={getattr(est, 'step_aggregation', '-')};"
                    ) if est.weight=='prob' else (
                        f"alpha={getattr(est, 'alpha', '-')};"
                        f"beta={getattr(est, 'beta', '-')};"
                    )
                },
            )


def bootstrap_std(ue, metric, ue_metric, num_runs: int = 1000, cfg=None, rng=None):
    ue = np.asarray(ue, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    valid = np.isfinite(ue) & np.isfinite(metric)
    ue = ue[valid]
    metric = metric[valid]
    if ue.size == 0:
        return 0.0
    if cfg is not None:
        num_runs = int(cfg.get("std_num_runs", num_runs))
        rng = np.random.default_rng(int(cfg.get("std_seed", 0)))
    rng = np.random.default_rng() if rng is None else rng
    mean_values = []
    for _ in range(num_runs):
        s = rng.integers(0, len(ue), size=len(ue))
        val = float(ue_metric(ue[s], metric[s]))
        if np.isfinite(val):
            mean_values.append(val)
    if not mean_values:
        return 0.0
    mean_values = np.array(mean_values)
    sorted_mean_values = np.sort(mean_values)
    lo = int(0.05 * len(sorted_mean_values))
    hi = int(0.95 * len(sorted_mean_values))
    trimmed = sorted_mean_values[lo:hi] if hi > lo else sorted_mean_values
    return float(trimmed.std())
