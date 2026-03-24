from typing import List, Optional
import logging

from data_service.typing.grpo_data import GRPOData, GlobalStepGRPOData
from utils.metrics import LocalMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_uniform(data: List[GRPOData]) -> bool:
    return all(data[0].reward_sum == data[i].reward_sum for i in range(1, len(data)))


def adaptive_grouping(
    all_data: List[List[GRPOData]],
    gpu_num: int,
    token_budget: int,
    max_micro_step_num: int,
    discard_clipped: bool = True,
    discard_aborted: bool = True,
    metrics: Optional[LocalMetrics] = None,
) -> GlobalStepGRPOData:
    global_step_data = GlobalStepGRPOData(
        gpu_num=gpu_num,
        token_budget=token_budget,
        max_micro_step_num=max_micro_step_num,
    )

    discard_because_uniform_reward = 0
    discard_because_no_response = 0
    discard_because_clipped = 0
    discard_because_aborted = 0
    discard_because_cannot_fit = 0
    discard_because_cannot_balance = 0

    flatten_data = []
    while all_data:
        if is_uniform(all_data[0]):
            discard_because_uniform_reward += len(all_data[0])
        else:
            flatten_data.extend(all_data[0])
        all_data.pop(0)

    flatten_data.sort(key=lambda x: x.length, reverse=True)
    filtered_data = []

    while flatten_data:
        if flatten_data[0].length > token_budget:
            discarded = flatten_data.pop(0)
            discard_because_cannot_fit += 1
            logger.debug(
                f"Discarding data that exceeds token budget: prompt_idx={discarded.prompt_idx}, response_idx={discarded.response_idx}, length={discarded.length}/{discarded.prompt_length}/{discarded.response_length}"
                f"conversation={discarded.conversation}, "
                f"prompt_token_ids={discarded.prompt_token_ids}, "
                f"response_token_ids={discarded.response_token_ids}"
            )
        elif flatten_data[0].response_length == 0:
            discarded = flatten_data.pop(0)
            discard_because_no_response += 1
            logger.error(
                f"Discarding data that has no response: prompt_idx={discarded.prompt_idx}, response_idx={discarded.response_idx}, length={discarded.length}/{discarded.prompt_length}/{discarded.response_length}, "
                f"conversation={discarded.conversation}, "
                f"prompt_token_ids={discarded.prompt_token_ids}, "
                f"response_token_ids={discarded.response_token_ids}"
            )
        elif flatten_data[0].stop_reason == "length" and discard_clipped:
            discarded = flatten_data.pop(0)
            discard_because_clipped += 1
            logger.debug(
                f"Discarding data that is clipped: prompt_idx={discarded.prompt_idx}, response_idx={discarded.response_idx}, length={discarded.length}/{discarded.prompt_length}/{discarded.response_length}, "
                f"conversation={discarded.conversation}, "
                f"prompt_token_ids={discarded.prompt_token_ids}, "
                f"response_token_ids={discarded.response_token_ids}"
            )
        elif flatten_data[0].stop_reason.endswith("abort") and discard_aborted:
            discarded = flatten_data.pop(0)
            discard_because_aborted += 1
            logger.debug(
                f"Discarding data that is aborted: prompt_idx={discarded.prompt_idx}, response_idx={discarded.response_idx}, length={discarded.length}/{discarded.prompt_length}/{discarded.response_length}, "
                f"conversation={discarded.conversation}, "
                f"prompt_token_ids={discarded.prompt_token_ids}, "
                f"response_token_ids={discarded.response_token_ids}"
            )
        else:
            filtered_data.append(flatten_data.pop(0))

    # Fit data into global step
    for data in filtered_data:
        if not global_step_data.fit_in(data):
            discard_because_cannot_fit += 1
            logger.debug(
                f"Discarding data that cannot fit in: prompt_idx={data.prompt_idx}, response_idx={data.response_idx}, length={data.length}/{data.prompt_length}/{data.response_length}"
            )
    if not global_step_data.is_empty:
        if not global_step_data.can_be_balanced():
            for discarded in global_step_data.discard_last_micro_step().flatten():
                discard_because_cannot_balance += 1
                logger.debug(
                    f"Discarding data that cannot be balanced: prompt_idx={discarded.prompt_idx}, response_idx={discarded.response_idx}, length={discarded.length}/{discarded.prompt_length}/{discarded.response_length}"
                )
        if not global_step_data.is_empty:
            global_step_data.balance()
            global_step_data.verify()

    if metrics:
        if global_step_data.micro_step_num > 0:
            metrics.add("Grouping/efficiency", global_step_data.efficiency)
        metrics.add("Grouping/seq_num", global_step_data.total_seq_num)
        metrics.add("Grouping/micro_step_num", global_step_data.micro_step_num)
        metrics.add("Grouping/max_budget", global_step_data.max_budget)
        metrics.add(
            "Grouping/equiv_step_num",
            global_step_data.total_effective_load / (token_budget * gpu_num),
        )
        metrics.add("Grouping/discard/uniform_reward", discard_because_uniform_reward)
        metrics.add("Grouping/discard/no_response", discard_because_no_response)
        metrics.add("Grouping/discard/clipped", discard_because_clipped)
        metrics.add("Grouping/discard/aborted", discard_because_aborted)
        metrics.add("Grouping/discard/cannot_fit", discard_because_cannot_fit)
        metrics.add("Grouping/discard/cannot_balance", discard_because_cannot_balance)

    return global_step_data


if __name__ == "__main__":
    import random

    def create_mock_grpo_data(
        prompt_idx: int, response_idx: int, length: Optional[int] = None
    ):
        if length is None:
            length = random.randint(5, 2048)

        from data_service.typing.message import Conversation, Message

        return GRPOData(
            prompt_idx=prompt_idx,
            response_idx=response_idx,
            conversation=Conversation(
                messages=[Message(role="system", content="Mock message")]
            ),
            solution="Mock solution",
            length=length,
            stop_reason="mock",
            ref_logprobs=[0.1, 0.2, 0.3],
            group_resp_token_num=1,
            group_seq_num=1,
        )

    def create_mock_list_grouped_grpo_data(
        num_groups: int, num_data_per_group: int
    ) -> List[List[GRPOData]]:
        all_data: List[List[GRPOData]] = []
        for i in range(num_groups):
            group_data = []
            for j in range(num_data_per_group):
                group_data.append(create_mock_grpo_data(i, j))
            all_data.append(group_data)
        return all_data

    while True:
        num_groups = random.randint(1, 16)
        num_data_per_group = random.randint(1, 16)
        gpu_num = random.randint(1, 8)
        token_budget = random.randint(30, 4096)
        max_micro_step_num = random.randint(1, 16)

        all_data = create_mock_list_grouped_grpo_data(num_groups, num_data_per_group)
        try:
            global_step_data = adaptive_grouping(
                all_data,
                gpu_num=gpu_num,
                token_budget=token_budget,
                max_micro_step_num=max_micro_step_num,
            )
            global_step_data.log()
        except Exception as e:
            print(e)
            raise e
            import pdb

            pdb.set_trace()
