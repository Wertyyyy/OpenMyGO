import asyncio
import logging
from typing import List, Optional

from .base import BaseOperation
from .rollout import RolloutOperation, RolloutInput
from .reward import RewardOperation
from .reduce import GroupParamsOperation
from .advantage import AdvantageOperation
from data_service.typing.grpo_data import GRPOData

logger = logging.getLogger(__name__)


class GRPOOperation(BaseOperation[RolloutInput, List[GRPOData]]):
    """Run rollout, reward, group reduction, and optional advantage for one prompt."""

    def __init__(
        self,
        rollout_op: RolloutOperation,
        reward_op: RewardOperation,
        group_params_op: GroupParamsOperation,
        advantage_op: Optional[AdvantageOperation] = None,
    ):
        self.rollout_op = rollout_op
        self.reward_op = reward_op
        self.group_params_op = group_params_op
        self.advantage_op = advantage_op

    async def process(self, rollout_input: RolloutInput) -> List[GRPOData]:
        reward_tasks = []
        rollout_stream = self.rollout_op(rollout_input)
        async for rollout_item in rollout_stream:
            reward_tasks.append(asyncio.create_task(self.reward_op(rollout_item)))

        if not reward_tasks:
            return []

        rewarded_items = await asyncio.gather(*reward_tasks, return_exceptions=True)
        batched_grpo_data = []
        for item in rewarded_items:
            if isinstance(item, Exception):
                logger.error(
                    f"Reward task failed for prompt {rollout_input.prompt_idx}: {item}"
                )
                continue
            batched_grpo_data.append(item)

        if not batched_grpo_data:
            logger.error(
                f"[Prompt {rollout_input.prompt_idx}] All reward tasks failed."
            )
            return []

        batched_grpo_data = await self.group_params_op(batched_grpo_data)

        if self.advantage_op:
            batched_grpo_data = await self.advantage_op(batched_grpo_data)

        return batched_grpo_data