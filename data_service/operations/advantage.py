from typing import List

import torch

from .base import BaseOperation
from data_service.typing.grpo_data import GRPOData


class AdvantageOperation(BaseOperation[List[GRPOData], List[GRPOData]]):
    """
    Operation for calculating advantages based on computed rewards.
    """

    def __init__(self, use_std: bool = True):
        self.use_std = use_std

    async def process(self, data: List[GRPOData]) -> List[GRPOData]:
        if not data:
            return data

        total_rewards = [item.reward_sum for item in data]
        mean_reward = torch.tensor(total_rewards).mean()

        if self.use_std:
            std_reward = torch.tensor(total_rewards).std()
            advantages = (
                (torch.tensor(total_rewards) - mean_reward) / (std_reward + 1e-4)
            ).view(-1)
        else:
            advantages = (torch.tensor(total_rewards) - mean_reward).view(-1)

        for item, advantage in zip(data, advantages):
            item.advantage = advantage.item()

        return data
