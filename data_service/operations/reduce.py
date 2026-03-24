from typing import List

from .base import BaseOperation
from data_service.typing.grpo_data import GRPOData


class GroupParamsOperation(BaseOperation[List[GRPOData], List[GRPOData]]):
    """Operation for calculating group-level params for each sample."""

    async def process(self, batched_grpo_data: List[GRPOData]) -> List[GRPOData]:
        if not batched_grpo_data:
            return batched_grpo_data

        group_response_token_sum = sum(d.response_length for d in batched_grpo_data)
        group_seq_num = len(batched_grpo_data)

        for data in batched_grpo_data:
            data.group_resp_token_sum = group_response_token_sum
            data.group_seq_num = group_seq_num

        return batched_grpo_data


class GlobalParamsOperation(BaseOperation[List[List[GRPOData]], List[List[GRPOData]]]):
    """Operation for calculating group/global params for each sample."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    async def process(
        self, filtered_data: List[List[GRPOData]]
    ) -> List[List[GRPOData]]:
        global_resp_token_sum = 0
        global_seq_num = 0
        global_group_num = 0

        for grouped_data in filtered_data:
            for data in grouped_data:
                global_resp_token_sum += data.response_length
                global_seq_num += 1
            global_group_num += 1

        for grouped_data in filtered_data:
            for data in grouped_data:
                data.global_resp_token_sum = global_resp_token_sum
                data.global_seq_num = global_seq_num
                data.global_group_num = global_group_num
                data.global_resp_max_len = self.max_tokens

        return filtered_data
