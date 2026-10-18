from abc import ABC, abstractmethod
from typing import List, Dict
from src.utils.process_prob_info import standard_probinfo_weight

class Estimator(ABC):
    """
    Abstract estimator class, which estimates the horizon uncertainty of a time series forecast.
    """

    @classmethod
    def get_subclass_dict(cls):
        """
        Returns a dictionary of all subclasses of the Estimator class.
        """
        return {subclass.__name__: subclass for subclass in cls.__subclasses__()}


    def __init__(self, **kwargs):
        """
        Use mean entropy for node scores, or constant weights for the ablation.
        """
        if "merge_way" in kwargs or "sim_way" in kwargs:
            raise ValueError("Deprecated keys 'merge_way/sim_way' are not supported. Use 'merge_policy/sim_tool'.")
        self.merge_policy = kwargs.get("merge_policy", "same-time")
        self.sim_tool = kwargs.get("sim_tool", "dtw")

        self.weight = kwargs.get("weight", "prob")
        self.token_level = kwargs.get("token_level", "neg entropy")
        self.step_aggregation = kwargs.get("step_aggregation", "average")
        self.beta = kwargs.get("beta", 1)

        if self.weight not in {"prob", ""}:
            raise ValueError("weight must be 'prob' or empty for the ablation")
        if self.token_level != "neg entropy" or self.step_aggregation != "average":
            raise ValueError("SGA uses entropy with average slice aggregation")

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the estimator.
        Class parameters which affect uncertainty estimates should also be included in the unique name to diversify between estimators.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, g_data) -> List[float]:
        """
        Abstract method. Calculates the uncertainty for each forecast trajectory in input statistics.

        Parameters:
            g_data (List[Dict]): list of graph data statistics. Each item is a dictionary containing
                all necessary information for uncertainty estimation.
        Returns:
            List[float]: list of uncertainty estimates for each forecast trajectory.
        """
        raise Exception("Not implemented")

    def _get_weight(self, node):
        '''
        Get the weight for a given node based on the specified weighting method.
        '''
        if self.weight == 'prob':
            v = sum(
                standard_probinfo_weight(prob_info_dict, self.token_level, way=self.step_aggregation)
                for prob_info_dict in node['prob_info_all']
            )/len(node['prob_info_all']) if 'prob_info_all' in node else 0
            return max(1 + v * self.beta, 0)
        else:
            return self.beta
