import flwr as fl
import numpy as np
import json
from functools import reduce
from logging import WARNING
from sklearn.neighbors import KernelDensity
# from sklearn.neighbors import KernelDensity
from typing import Callable, Dict, List, Optional, Tuple, Union
from typing import Optional, Callable, Dict, Tuple, List, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# 假设这是拆分后的权重和对应的形状信息
best_weights = [
    np.random.rand(64, 1, 3),  # Shape: (64, 1, 3)
    np.random.rand(64),        # Shape: (64,)
    np.random.rand(32, 64, 3), # Shape: (32, 64, 3)
    np.random.rand(32),        # Shape: (32,)
    np.random.rand(1, 384),    # Shape: (1, 384)
    np.random.rand(1),         # Shape: (1,)
]

# 存储每个权重的形状信息
weights_shapes = [w.shape for w in best_weights]

# 还原的函数
def restore_weights(packed_weights, shapes):
    restored_weights = []
    for weight, shape in zip(packed_weights, shapes):
        restored_weights.append(weight.reshape(shape))
    return restored_weights


class Dis(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace


    def aggregate_fit( 
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and filter based on jsd_value."""

        # 如果没有结果，则直接返回
        if not results:
            return None, {}

        # 如果有失败客户端且不接受失败，直接返回
        if not self.accept_failures and failures:
            return None, {}

        # 初始化用于聚合的结果和 jsd_value 列表
        weights_results = []
        jsd_values = []

        # 提取每个客户端的模型参数、jsd_value 和最佳权重
        for client, fit_res in results:
            # 确保从 metrics 中访问 jsd_value
            if 'current_jsd_value' in fit_res.metrics:
                jsd_value = fit_res.metrics['current_jsd_value']
                jsd_values.append(jsd_value)
                
                # 拆分 parameters
                parameters = parameters_to_ndarrays(fit_res.parameters)
                
                # 假设我们将参数拆分为前半部分和后半部分
                mid_point = len(parameters) // 2
                local_parameters = parameters[:mid_point]
                best_weights = parameters[mid_point:]

                local_parameters = restore_weights(local_parameters, weights_shapes)
                best_weights = restore_weights(best_weights, weights_shapes)

                # 将客户端的参数、对应的样本数和 jsd_value 保存
                weights_results.append((local_parameters, best_weights, fit_res.num_examples, jsd_value))
            else:
                log(WARNING, f"Client {client} did not return 'current_jsd_value'.")

        # 计算所有客户端的 jsd_value 平均值
        average_jsd_value = sum(jsd_values) / len(jsd_values)
        print(f"Average JSD value: {average_jsd_value}")

        # 初始化聚合列表
        results = []

        # 根据 jsd_value 筛选出结果
        for (local_parameters, best_weights, num_examples, jsd_value) in weights_results:
            if jsd_value < average_jsd_value:
                # 当 jsd_value 小于或等于平均值时，使用当前权重
                results.append((local_parameters, num_examples))
            else:
                # 当 jsd_value 大于等于平均值时，使用最佳权重
                results.append((best_weights, num_examples))

        # 如果没有符合条件的客户端，则返回 None
        if not results:
            return None, {}

        # 聚合权重
        if self.inplace:
            aggregated_ndarrays = self.aggregate_inplace(results)
        else:
            # aggregated_ndarrays = self.aggregate(results) 
            aggregated_ndarrays = self.aggregate_flame_f(results) 

        # 转换聚合后的参数
        final_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return final_parameters, metrics_aggregated

    
    def aggregate_inplace(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        """Compute in-place weighted average."""
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

        return params
    
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime


    def aggregate_flame_f(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in results)
        # 使用列表推导式来计算加权系数
        weight_scalling_factor_list = [num_examples / num_examples_total for (_, num_examples) in results]


        # get last feature layer
        w_last_layer = []
        b_last_layer = []

        for i in range(len(results)):
            w_last_layer.append(np.array(results[i][0][-4]))        # collect weight
            b_last_layer.append(np.array(results[i][0][-3]))        # collect bias

        # w_last_layer flattened as vectors
        w_last_layer = np.array(w_last_layer).reshape(len(w_last_layer), -1)
        b_last_layer = np.array(b_last_layer).reshape(len(b_last_layer), -1)

        # using KDE get the kernel density of last layers
        kde_w = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(w_last_layer, sample_weight=weight_scalling_factor_list)
        kde_b = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(b_last_layer, sample_weight=weight_scalling_factor_list)

        # sample m samples and average, then obtain a new last layer for the global model
        w_last_layer_new = np.mean(kde_w.sample(len(w_last_layer)), axis=0)
        w_last_layer_new = w_last_layer_new.reshape([32, 64, 3])
        
        b_last_layer_new = np.mean(kde_b.sample(len(b_last_layer)), axis=0)
        b_last_layer_new = b_last_layer_new.reshape([32,])


        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        # replace last layers
        weights_prime[-4] = w_last_layer_new # for last feature layer's weight
        weights_prime[-3] = b_last_layer_new # for last feature layer's bias

        return weights_prime