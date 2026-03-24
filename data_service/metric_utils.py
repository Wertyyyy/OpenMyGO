import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from data_service.typing.grpo_data import GRPOData
from utils.metrics import LocalMetrics

logger = logging.getLogger(__name__)


def _extract_time_window(data_list: List[GRPOData]) -> Optional[Tuple[float, float]]:
    starts = [data.start_time for data in data_list if data.start_time is not None]
    ends = [data.end_time for data in data_list if data.end_time is not None]
    if not starts or not ends:
        return None

    start_time = min(starts)
    end_time = max(ends)
    if end_time <= start_time:
        return None
    return start_time, end_time


def _completion_time_ratio(
    data_list: List[GRPOData], start_time: float, total_time: float, ratio: float
) -> Optional[float]:
    if total_time <= 0:
        return None

    valid_end_times = sorted(
        data.end_time for data in data_list if data.end_time is not None
    )
    if not valid_end_times:
        return None

    target_idx = max(0, min(math.ceil(len(valid_end_times) * ratio) - 1, len(valid_end_times) - 1))
    completion_end_time = valid_end_times[target_idx]
    elapsed = completion_end_time - start_time
    elapsed = max(0.0, min(elapsed, total_time))
    return elapsed / total_time


def compute_rollout_metrics(filtered_data: List[List[GRPOData]]) -> LocalMetrics:
    metrics = LocalMetrics()

    all_grpo_data: List[GRPOData] = []
    for grouped_data in filtered_data:
        all_grpo_data.extend(grouped_data)

    all_grpo_data = [data for data in all_grpo_data if data.label == "Rollout"]

    if not all_grpo_data:
        return metrics

    prompt_lengths = [len(data.prompt_token_ids) for data in all_grpo_data]
    response_lengths = [len(data.response_token_ids) for data in all_grpo_data]

    stop_count = 0
    length_count = 0
    abort_count = 0

    response_lengths_stop: List[int] = []
    response_lengths_no_abort: List[int] = []

    for data in all_grpo_data:
        response_len = len(data.response_token_ids)
        stop_reason = data.stop_reason

        if stop_reason == "stop":
            stop_count += 1
            response_lengths_stop.append(response_len)
            response_lengths_no_abort.append(response_len)
        elif stop_reason == "length":
            length_count += 1
            response_lengths_no_abort.append(response_len)
        elif stop_reason.endswith("abort"):
            abort_count += 1
        else:
            stop_count += 1
            response_lengths_stop.append(response_len)
            response_lengths_no_abort.append(response_len)

    total_count = len(all_grpo_data)
    if total_count > 0:
        metrics.add("Data/termination/stop_count", stop_count)
        metrics.add("Data/termination/stop_ratio", stop_count / total_count)
        metrics.add("Data/termination/length_count", length_count)
        metrics.add("Data/termination/length_ratio", length_count / total_count)
        metrics.add("Data/termination/abort_count", abort_count)
        metrics.add("Data/termination/abort_ratio", abort_count / total_count)

    if prompt_lengths:
        metrics.add("Data/prompt/min", min(prompt_lengths))
        metrics.add("Data/prompt/max", max(prompt_lengths))
        metrics.add("Data/prompt/mean", statistics.mean(prompt_lengths))

    if response_lengths:
        metrics.add("Data/response/min", min(response_lengths))
        metrics.add("Data/response/max", max(response_lengths))
        metrics.add("Data/response/mean", statistics.mean(response_lengths))

    if response_lengths_stop:
        metrics.add("Data/response/max_stop", max(response_lengths_stop))
        metrics.add(
            "Data/response/mean_stop", statistics.mean(response_lengths_stop)
        )

    if response_lengths_no_abort:
        metrics.add("Data/response/max_no_abort", max(response_lengths_no_abort))
        metrics.add(
            "Data/response/mean_no_abort",
            statistics.mean(response_lengths_no_abort),
        )

    reward_names = (
        list(all_grpo_data[0].rewards.keys()) if all_grpo_data[0].rewards else []
    )
    for reward_name in reward_names:
        reward_values = [data.rewards[reward_name] for data in all_grpo_data]
        if reward_values:
            metrics.add(f"Reward/{reward_name}/min", min(reward_values))
            metrics.add(f"Reward/{reward_name}/max", max(reward_values))
            metrics.add(
                f"Reward/{reward_name}/mean", statistics.mean(reward_values)
            )
            if len(reward_values) > 1:
                metrics.add(
                    f"Reward/{reward_name}/std", statistics.stdev(reward_values)
                )

    advantages = [
        data.advantage for data in all_grpo_data if data.advantage is not None
    ]
    if advantages:
        metrics.add("Reward/advantage/min", min(advantages))
        metrics.add("Reward/advantage/max", max(advantages))
        metrics.add("Reward/advantage/mean", statistics.mean(advantages))
        if len(advantages) > 1:
            metrics.add("Reward/advantage/std", statistics.stdev(advantages))

    uniform_reward_groups = 0
    zero_reward_groups = 0
    full_reward_groups = 0
    total_groups = len(filtered_data)

    for grouped_data in filtered_data:
        group_rewards = [data.reward_sum for data in grouped_data]
        unique_values = list(set(group_rewards))
        if len(unique_values) == 1:
            uniform_reward_groups += 1
            if unique_values[0] == 0:
                zero_reward_groups += 1
            else:
                full_reward_groups += 1

    if total_groups > 0:
        metrics.add(
            "Reward/validity/uniform_ratio", uniform_reward_groups / total_groups
        )
        metrics.add("Reward/validity/zero_ratio", zero_reward_groups / total_groups)
        metrics.add("Reward/validity/full_ratio", full_reward_groups / total_groups)

    time_window = _extract_time_window(all_grpo_data)
    generation_time = None
    if time_window is not None:
        start_time, end_time = time_window
        generation_time = end_time - start_time
        metrics.add("Data/performance/time", generation_time)

        completion_50_ratio = _completion_time_ratio(
            all_grpo_data, start_time, generation_time, ratio=0.5
        )
        if completion_50_ratio is not None:
            metrics.add("Data/performance/50pct_time_ratio", completion_50_ratio)

        completion_90_ratio = _completion_time_ratio(
            all_grpo_data, start_time, generation_time, ratio=0.9
        )
        if completion_90_ratio is not None:
            metrics.add("Data/performance/90pct_time_ratio", completion_90_ratio)

    total_response_tokens = sum(response_lengths)

    if generation_time is not None and generation_time > 0:
        response_tokens_per_sec = total_response_tokens / generation_time
        metrics.add("Data/performance/tps", response_tokens_per_sec)

    return metrics


def compute_evaluation_metrics(
    dataset_name: str,
    results: List[List[GRPOData]],
) -> LocalMetrics:
    """Compute evaluation metrics from GRPOData results."""
    metrics = LocalMetrics()

    if not results:
        return metrics

    all_responses: List[GRPOData] = []
    for batch in results:
        all_responses.extend(batch)

    if not all_responses:
        return metrics

    total_samples = len(results)
    total_responses = len(all_responses)

    response_token_lengths = [
        len(result.response_token_ids) for result in all_responses
    ]

    prefix = f"Eval-{dataset_name.upper()}"

    stop_count = 0
    length_count = 0
    abort_count = 0

    response_lengths_stop: List[int] = []
    response_lengths_no_abort: List[int] = []

    for result in all_responses:
        response_len = len(result.response_token_ids)
        stop_reason = result.stop_reason

        if stop_reason == "stop":
            stop_count += 1
            response_lengths_stop.append(response_len)
            response_lengths_no_abort.append(response_len)
        elif stop_reason == "length":
            length_count += 1
            response_lengths_no_abort.append(response_len)
        elif stop_reason == "abort":
            abort_count += 1
        else:
            stop_count += 1
            response_lengths_stop.append(response_len)
            response_lengths_no_abort.append(response_len)

    if total_responses > 0:
        metrics.add(f"{prefix}/termination/stop_ratio", stop_count / total_responses)
        metrics.add(
            f"{prefix}/termination/length_ratio", length_count / total_responses
        )
        metrics.add(f"{prefix}/termination/abort_ratio", abort_count / total_responses)

    if response_token_lengths:
        metrics.add(f"{prefix}/response/min", min(response_token_lengths))
        metrics.add(f"{prefix}/response/max", max(response_token_lengths))
        metrics.add(f"{prefix}/response/mean", statistics.mean(response_token_lengths))

    if response_lengths_stop:
        metrics.add(f"{prefix}/response/max_stop", max(response_lengths_stop))
        metrics.add(
            f"{prefix}/response/mean_stop", statistics.mean(response_lengths_stop)
        )

    if response_lengths_no_abort:
        metrics.add(f"{prefix}/response/max_no_abort", max(response_lengths_no_abort))
        metrics.add(
            f"{prefix}/response/mean_no_abort",
            statistics.mean(response_lengths_no_abort),
        )

    eval_time_window = _extract_time_window(all_responses)
    evaluation_time = None
    if eval_time_window is not None:
        evaluation_time = eval_time_window[1] - eval_time_window[0]
        metrics.add(f"{prefix}/performance/time", evaluation_time)
    total_response_tokens = sum(response_token_lengths)

    if evaluation_time is not None and evaluation_time > 0:
        response_tokens_per_sec = total_response_tokens / evaluation_time
        samples_per_sec = total_samples / evaluation_time

        metrics.add(f"{prefix}/performance/tps", response_tokens_per_sec)
        metrics.add(f"{prefix}/performance/sps", samples_per_sec)

    if all_responses and all_responses[0].rewards:
        reward_names = list(all_responses[0].rewards.keys())
        for reward_name in reward_names:
            reward_values = [result.rewards[reward_name] for result in all_responses]
            if reward_values:
                metrics.add(
                    f"{prefix}/reward-{reward_name}/mean",
                    statistics.mean(reward_values),
                )
                if len(reward_values) > 1:
                    metrics.add(
                        f"{prefix}/reward-{reward_name}/std",
                        statistics.stdev(reward_values),
                    )

    return metrics


def compute_overall_evaluation_metrics(
    dataset_results_list: List[Dict[str, Any]],
) -> LocalMetrics:
    """Compute overall weighted average metrics across all evaluation datasets."""
    metrics = LocalMetrics()

    if not dataset_results_list:
        return metrics

    prefix = "Eval-OVERALL"

    all_rewards: Dict[str, List[float]] = {}
    all_response_lengths: List[int] = []

    stop_count = 0
    length_count = 0
    abort_count = 0

    response_lengths_stop: List[int] = []
    response_lengths_no_abort: List[int] = []

    total_samples = 0
    total_responses = 0

    for dataset_info in dataset_results_list:
        results = dataset_info["results"]
        total_samples += len(results)

        for batch in results:
            for result in batch:
                total_responses += 1
                response_len = len(result.response_token_ids)
                all_response_lengths.append(response_len)
                stop_reason = result.stop_reason

                if stop_reason == "stop":
                    stop_count += 1
                    response_lengths_stop.append(response_len)
                    response_lengths_no_abort.append(response_len)
                elif stop_reason == "length":
                    length_count += 1
                    response_lengths_no_abort.append(response_len)
                elif stop_reason == "abort":
                    abort_count += 1
                else:
                    stop_count += 1
                    response_lengths_stop.append(response_len)
                    response_lengths_no_abort.append(response_len)

                for reward_name, reward_value in result.rewards.items():
                    if reward_name not in all_rewards:
                        all_rewards[reward_name] = []
                    all_rewards[reward_name].append(reward_value)

    if total_responses > 0:
        metrics.add(f"{prefix}/termination/stop_ratio", stop_count / total_responses)
        metrics.add(
            f"{prefix}/termination/length_ratio", length_count / total_responses
        )
        metrics.add(f"{prefix}/termination/abort_ratio", abort_count / total_responses)

    if all_response_lengths:
        metrics.add(f"{prefix}/response/min", min(all_response_lengths))
        metrics.add(f"{prefix}/response/max", max(all_response_lengths))
        metrics.add(f"{prefix}/response/mean", statistics.mean(all_response_lengths))

    if response_lengths_stop:
        metrics.add(f"{prefix}/response/max_stop", max(response_lengths_stop))
        metrics.add(
            f"{prefix}/response/mean_stop", statistics.mean(response_lengths_stop)
        )

    if response_lengths_no_abort:
        metrics.add(f"{prefix}/response/max_no_abort", max(response_lengths_no_abort))
        metrics.add(
            f"{prefix}/response/mean_no_abort",
            statistics.mean(response_lengths_no_abort),
        )

    for reward_name, reward_values in all_rewards.items():
        if reward_values:
            metrics.add(
                f"{prefix}/reward-{reward_name}/mean", statistics.mean(reward_values)
            )
            if len(reward_values) > 1:
                metrics.add(
                    f"{prefix}/reward-{reward_name}/std",
                    statistics.stdev(reward_values),
                )

    all_responses: List[GRPOData] = []
    for dataset_info in dataset_results_list:
        for batch in dataset_info["results"]:
            all_responses.extend(batch)

    total_eval_time = None
    overall_time_window = _extract_time_window(all_responses)
    if overall_time_window is not None:
        total_eval_time = overall_time_window[1] - overall_time_window[0]
        metrics.add(f"{prefix}/performance/time", total_eval_time)

    if total_eval_time is not None and total_eval_time > 0 and all_response_lengths:
        total_response_tokens = sum(all_response_lengths)
        response_tokens_per_sec = total_response_tokens / total_eval_time
        samples_per_sec = total_samples / total_eval_time

        metrics.add(f"{prefix}/performance/tps", response_tokens_per_sec)
        metrics.add(f"{prefix}/performance/sps", samples_per_sec)

    logger.info(
        "Overall evaluation metrics computed: "
        f"{len(dataset_results_list)} datasets, {total_samples} samples, "
        f"stop={stop_count}, length={length_count}, abort={abort_count}"
    )

    return metrics
