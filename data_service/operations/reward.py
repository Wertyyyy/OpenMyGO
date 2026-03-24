import logging
import inspect
from typing import Dict, Any

from .base import BaseOperation
from data_service.typing.grpo_data import GRPOData

logger = logging.getLogger(__name__)

class RewardOperation(BaseOperation[GRPOData, GRPOData]):
    """
    Operation for calculating rewards for a single GRPOData item.
    """
    def __init__(self, reward_fns: Dict[str, Any]):
        self.reward_fns = reward_fns

    async def process(self, data: GRPOData) -> GRPOData:
        for reward_name, reward_fn in self.reward_fns.items():
            try:
                # Check if reward function is async
                if inspect.iscoroutinefunction(reward_fn.__call__):
                    result = await reward_fn(data)
                else:
                    result = reward_fn(data)
                data.rewards[reward_name], data.reward_reasons[reward_name] = result
            except Exception as e:
                logger.error(f"Error calculating reward '{reward_name}': {e}")
                data.rewards[reward_name] = 0.0
                data.reward_reasons[reward_name] = f"Error: {e}"
        return data

