import logging
import copy
import time
from typing import Dict, Any, Union, AsyncGenerator

from pydantic import BaseModel

from .base import BaseStreamOperation
from data_service.typing.message import Conversation
from data_service.typing.grpo_data import GRPOData
from vllm_service.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class RolloutInput(BaseModel):
    conversation: Conversation
    solution: str
    prompt_idx: Union[str, int]
    label: str = "Rollout"


class RolloutOperation(BaseStreamOperation[RolloutInput, AsyncGenerator[GRPOData, None]]):
    """
    Operation for generating completions using vLLM model.
    """

    def __init__(
        self,
        vllm_client: VLLMClient,
        processor: Any,
        sampling_params: Dict[str, Any],
        max_length: int,
    ):
        self.vllm_client = vllm_client
        self.processor = processor
        self.sampling_params = sampling_params
        self.max_length = max_length

    async def process(self, data: RolloutInput) -> AsyncGenerator[GRPOData, None]:
        seen_response_indices = set()
        try:
            start_time = time.time()

            async for event in self.vllm_client.generate(
                conversation=data.conversation,
                sampling_params=self.sampling_params,
            ):
                rollout_idx = int(event["index"])
                if rollout_idx in seen_response_indices:
                    logger.warning(
                        f"[Prompt {data.prompt_idx}] Duplicate streamed rollout index {rollout_idx} skipped"
                    )
                    continue
                seen_response_indices.add(rollout_idx)

                rollout = event["completion"]

                # Normalize specific tags
                rollout = rollout.replace("<think>", " ").replace("</think>", " ")

                conversation_ = copy.deepcopy(data.conversation)
                conversation_.add_message(role="assistant", content=rollout)

                try:
                    prompt_token_ids, response_token_ids = (
                        self.processor.get_prompt_response_token_ids(conversation_)
                    )
                except Exception as e:
                    logger.error(
                        f"[Prompt {data.prompt_idx}, Rollout {rollout_idx}] Processor error: {e}"
                    )
                    continue

                if len(prompt_token_ids) + len(response_token_ids) > self.max_length:
                    logger.warning(
                        f"[Prompt {data.prompt_idx}, Rollout {rollout_idx}] Response dropped due to exceeding max total length."
                    )
                    continue

                yield GRPOData(
                    prompt_idx=data.prompt_idx,
                    response_idx=rollout_idx,
                    conversation=conversation_,
                    solution=data.solution,
                    stop_reason=event["finish_reason"],
                    start_time=start_time,
                    end_time=time.time(),
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    label=data.label,
                )

        except Exception as e:
            logger.error(f"[Prompt {data.prompt_idx}] VLLM generation failed: {e}")
            return
