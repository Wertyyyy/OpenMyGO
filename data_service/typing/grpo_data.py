from typing import List, Optional, Union, Tuple, Dict, Any, Literal
import logging

import torch
from pydantic import BaseModel, Field, field_validator, field_serializer

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GRPOData(BaseModel):
    prompt_idx: Union[str, int] = Field(...)
    response_idx: int = Field(...)
    label: Optional[str] = Field(None)

    start_time: Optional[float] = Field(None)
    end_time: Optional[float] = Field(None)

    conversation: Conversation = Field(...)
    solution: str = Field(...)
    stop_reason: str = Field(...)

    response_token_ids: List[int] = Field(...)
    prompt_token_ids: List[int] = Field(...)

    ref_logprobs: Optional[torch.Tensor] = Field(None)
    pol_logprobs: Optional[torch.Tensor] = Field(None)
    per_token_entropy: Optional[torch.Tensor] = Field(None)
    per_token_kl: Optional[torch.Tensor] = Field(None)
    per_token_loss: Optional[torch.Tensor] = Field(None)

    group_resp_token_sum: Optional[int] = Field(None)
    group_seq_num: Optional[int] = Field(None)
    global_resp_token_sum: Optional[int] = Field(None)
    global_seq_num: Optional[int] = Field(None)
    global_group_num: Optional[int] = Field(None)
    global_resp_max_len: Optional[int] = Field(None)

    rewards: Optional[Dict[str, float]] = Field(default_factory=dict)
    reward_reasons: Optional[Dict[str, str]] = Field(default_factory=dict)
    advantage: Optional[float] = Field(None)

    class Config:
        arbitrary_types_allowed = True

    @field_validator(
        "ref_logprobs",
        "pol_logprobs",
        "per_token_entropy",
        "per_token_kl",
        "per_token_loss",
        mode="before",
    )
    @classmethod
    def validate_tensor_fields(cls, v):
        if v is None:
            return v
        elif isinstance(v, torch.Tensor):
            if v.dim() != 1:
                raise ValueError(f"Field must be 1D tensor, got {v.dim()}D tensor")
            return v
        elif isinstance(v, list):
            if all(isinstance(x, (int, float)) for x in v):
                return torch.tensor(v, dtype=torch.float32, device=torch.device("cpu"))
            raise ValueError("List must contain only numbers, got mixed types")
        else:
            raise ValueError(
                f"Field must be None, List[float] or torch.Tensor, got {type(v)}"
            )

    @field_serializer(
        "ref_logprobs",
        "pol_logprobs",
        "per_token_entropy",
        "per_token_kl",
        "per_token_loss",
    )
    def serialize_tensor_fields(self, value):
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        return value

    @property
    def length(self) -> int:
        return len(self.response_token_ids) + len(self.prompt_token_ids)

    @property
    def response_length(self) -> int:
        return len(self.response_token_ids)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def reward_sum(self) -> float:
        return sum(self.rewards.values())

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Data {idx} (prompt_idx={self.prompt_idx}, response_idx={self.response_idx}, length={self.length}/{self.prompt_length}/{self.response_length}, stop_reason={self.stop_reason})"
        )


class BatchedGRPOData(BaseModel):
    token_budget: int = Field(..., description="Token budget for the batch")
    data: List[GRPOData] = Field(
        default_factory=list, description="List of GRPO data items"
    )

    @property
    def longest_seq_len(self) -> int:
        """The length of the longest sequence in the batch."""
        return max([item.length for item in self.data], default=0)

    @property
    def seq_num(self) -> int:
        """The number of sequences in the batch."""
        return len(self.data)

    @property
    def extra_seq_num(self) -> int:
        """The number of extra sequences in the batch."""
        return max(self.seq_num - 1, 0)

    @property
    def current_load(self) -> int:
        """The current load of the batch."""
        return self.longest_seq_len * self.seq_num

    @property
    def effective_load(self) -> int:
        """The effective load of the batch."""
        return sum([item.length for item in self.data])

    @property
    def efficiency(self) -> float:
        """The efficiency of the batch."""
        assert self.data, "Batch is empty"
        return self.effective_load / self.token_budget

    def fit_in(self, data: GRPOData, add_to_batch: bool = True) -> bool:
        """Fit an item into the batch."""
        if (
            max(self.longest_seq_len, data.length) * (self.seq_num + 1)
            > self.token_budget
        ):
            return False
        if add_to_batch:
            self.data.append(data)
            self.data.sort(key=lambda x: x.length, reverse=True)
        return True

    def pop(self) -> Optional[GRPOData]:
        assert self.seq_num > 0, "Batch is empty"
        return self.data.pop()

    def verify(self):
        """Verify the batch."""
        assert self.seq_num > 0, "Batch is empty"
        assert self.current_load <= self.token_budget, (
            f"Batch is over budget: {self.current_load} > {self.token_budget}"
        )

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Batch {idx} (seq_num={self.seq_num}, token_budget={self.token_budget}, current_load={self.current_load}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)

    def set_data_fields(self, field_name: str, value: List[Any]):
        for item, v in zip(self.data, value):
            setattr(item, field_name, v)

    def get_data_fields(self, field_name: str, device=None) -> List:
        if device is None:
            # HACK: Get the device of the current process
            from accelerate import PartialState

            state = PartialState()
            device = state.device

        result = []
        for item in self.data:
            value = getattr(item, field_name)
            if hasattr(value, "to") and hasattr(value, "device"):
                value = value.to(device)
            result.append(value)
        return result

    def denominator_per_data(
        self, avg_type: Literal["local", "group", "global"]
    ) -> List[int]:
        if avg_type == "local":
            return [data.response_length * data.global_seq_num for data in self.data]
        elif avg_type == "group":
            return [
                data.group_resp_token_sum * data.global_group_num for data in self.data
            ]
        elif avg_type == "global":
            return [data.global_resp_token_sum for data in self.data]
        elif avg_type == "global_max":
            return [
                data.global_resp_max_len * data.global_seq_num for data in self.data
            ]

    def sum_with_denominator(
        self, tensor_list, avg_type: Literal["local", "group", "global"]
    ) -> Optional[torch.Tensor]:
        tensor_list = [t for t in tensor_list if t is not None]
        if not tensor_list:
            return None

        return sum(
            [
                torch.sum(t / d)
                for t, d in zip(tensor_list, self.denominator_per_data(avg_type))
            ]
        )

    def detach(self):
        """Detach all tensors from computation graph to free memory."""
        tensor_fields = [
            "ref_logprobs",
            "pol_logprobs",
            "per_token_entropy",
            "per_token_kl",
            "per_token_loss",
        ]
        for data_item in self.data:
            for field_name in tensor_fields:
                tensor = getattr(data_item, field_name, None)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Detach tensor in-place
                    setattr(data_item, field_name, tensor.detach())


class MicroStepGRPOData(BaseModel):
    gpu_num: int = Field(..., description="Number of GPUs")
    token_budget: int = Field(..., description="Token budget per batch")
    data: List[BatchedGRPOData] = Field(
        default_factory=list, description="List of batched GRPO data"
    )

    def model_post_init(self, __context):
        """Initialize empty batches to match GPU count."""
        while len(self.data) < self.gpu_num:
            self.data.append(BatchedGRPOData(token_budget=self.token_budget))

    def fit_in(self, data: GRPOData, add_to_batch: bool = True) -> bool:
        """Fit an item into the batch."""
        for batch in self.data:
            if batch.fit_in(data, add_to_batch):
                return True
        return False

    @property
    def total_seq_num(self) -> int:
        """The total number of sequences in the batch."""
        return sum(batch.seq_num for batch in self.data)

    @property
    def total_extra_seq_num(self) -> int:
        """The total number of extra sequences in the batch."""
        return sum(batch.extra_seq_num for batch in self.data)

    @property
    def seq_needed_num(self) -> int:
        if self.total_seq_num >= self.gpu_num:
            return 0
        return self.gpu_num - self.total_seq_num

    @property
    def total_effective_load(self) -> int:
        """The effective load of the batch."""
        return sum(batch.effective_load for batch in self.data)

    @property
    def max_load(self) -> int:
        """The max load of the batch."""
        return max(batch.current_load for batch in self.data)

    @property
    def efficiency(self) -> float:
        """The efficiency of the batch."""
        assert self.max_load > 0, "Max load should be greater than 0"
        return self.total_effective_load / (self.gpu_num * self.max_load)

    def pop(self) -> GRPOData:
        """Pop a GRPOData item from the batches, starting from the last batch."""
        for batch in reversed(self.data):
            if batch.seq_num > 0:
                return batch.pop()
        assert False

    def flatten(self) -> List[GRPOData]:
        """Get all GRPOData items from all batches."""
        result = []
        for batch in self.data:
            result.extend(batch.data)
        return result

    def verify(self):
        assert len(self.data) == self.gpu_num, (
            f"Batch size does not match GPU number: {len(self.data)} != {self.gpu_num}"
        )

        for batch in self.data:
            batch.verify()

    def _find_max_load_batch(self) -> Tuple[int, BatchedGRPOData]:
        """Find the index of the batch with maximum load that can provide sequences."""
        valid_batches = [
            (idx, batch) for idx, batch in enumerate(self.data) if batch.seq_num > 1
        ]
        return max(valid_batches, key=lambda x: x[1].current_load, default=(None, None))

    def _find_min_load_batch(self) -> Tuple[int, BatchedGRPOData]:
        """Find the index of the batch with minimum load."""
        return min(enumerate(self.data), key=lambda x: x[1].current_load)

    def _will_improve_balance(
        self, source_batch: BatchedGRPOData, target_batch: BatchedGRPOData
    ) -> bool:
        assert source_batch.seq_num > 1
        item_to_move = source_batch.data[-1]

        current_max_load = self.max_load

        source_new_longest = source_batch.data[0].length
        source_new_load = source_new_longest * (source_batch.seq_num - 1)

        target_new_longest = max(target_batch.longest_seq_len, item_to_move.length)
        target_new_load = target_new_longest * (target_batch.seq_num + 1)

        others_max_load = max(
            [
                b.current_load
                for b in self.data
                if b != source_batch and b != target_batch
            ],
            default=0,
        )

        new_max_load = max(source_new_load, target_new_load, others_max_load)
        return new_max_load < current_max_load

    def _execute_move(
        self, source_batch: BatchedGRPOData, target_batch: BatchedGRPOData
    ):
        item = source_batch.data.pop()
        target_batch.fit_in(item)
        return item

    def balance(self):
        assert self.total_seq_num >= self.gpu_num

        initial_max_load = self.max_load
        while True:
            source_idx, source_batch = self._find_max_load_batch()
            target_idx, target_batch = self._find_min_load_batch()

            if source_idx is None or source_idx == target_idx:
                break

            if target_batch.seq_num == 0:
                item = self._execute_move(source_batch, target_batch)
                logger.debug(
                    f"Move item: {item.prompt_idx}, {item.response_idx}, from batch {source_idx} to {target_idx}, Reason: Target batch is empty"
                )
            elif self._will_improve_balance(source_batch, target_batch):
                eff_before = self.efficiency
                item = self._execute_move(source_batch, target_batch)
                eff_after = self.efficiency
                assert eff_after >= eff_before
                logger.debug(
                    f"Move item: {item.prompt_idx}, {item.response_idx}, from batch {source_idx} to {target_idx}, Reason: Will improve balance, Efficiency before: {eff_before}, after: {eff_after}"
                )
            else:
                break

        assert self.max_load <= initial_max_load

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}MicroStep {idx} (gpu_num={self.gpu_num}, seq_num={self.total_seq_num}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)


class GlobalStepGRPOData(BaseModel):
    token_budget: int = Field(..., description="Token budget per batch")
    gpu_num: int = Field(..., description="Number of GPUs")
    max_micro_step_num: int = Field(
        ..., gt=0, description="Maximum number of micro steps"
    )
    data: List[MicroStepGRPOData] = Field(
        default_factory=list, description="List of micro step data"
    )

    @property
    def is_empty(self) -> bool:
        return len(self.data) == 0

    @property
    def micro_step_num(self):
        return len(self.data)

    @property
    def total_seq_num(self) -> int:
        """The total number of sequences in the batch."""
        return sum(micro_step.total_seq_num for micro_step in self.data)

    @property
    def total_batch_num(self) -> int:
        """The total number of batches in the batch."""
        return self.micro_step_num * self.gpu_num

    @property
    def total_effective_load(self) -> int:
        """The total number of effective tokens in the batch."""
        return sum(micro_step.total_effective_load for micro_step in self.data)

    @property
    def max_budget(self) -> int:
        max_budget = 0
        for micro_step in self.data:
            max_budget = max(max_budget, micro_step.max_load)
        return max_budget

    @property
    def efficiency(self) -> float:
        assert len(self.data) != 0, "No micro steps available"
        micro_step_efficiency = [micro_step.efficiency for micro_step in self.data]
        return sum(micro_step_efficiency) / len(micro_step_efficiency)

    @property
    def extra_seq_num_from_previous_micro_steps(self) -> int:
        return sum(micro_step.total_extra_seq_num for micro_step in self.data[:-1])

    @property
    def seq_needed_num_from_last_micro_step(self) -> int:
        assert len(self.data) != 0, "No micro steps available"
        return self.data[-1].seq_needed_num

    def create_micro_step(self) -> bool:
        if self.micro_step_num >= self.max_micro_step_num:
            return False
        self.data.append(
            MicroStepGRPOData(gpu_num=self.gpu_num, token_budget=self.token_budget)
        )
        return True

    def fit_in(self, data: GRPOData) -> bool:
        assert data.length <= self.token_budget

        for micro_step in self.data:
            if micro_step.fit_in(data):
                return True
        if self.create_micro_step():
            return self.fit_in(data)
        return False

    def _pop_from_previous_micro_steps(self) -> GRPOData:
        for micro_step in reversed(self.data[:-1]):
            if micro_step.total_extra_seq_num > 0:
                return micro_step.pop()
        assert False, "Should not reach here if assertions above are correct"

    def can_be_balanced(self) -> bool:
        return (
            self.extra_seq_num_from_previous_micro_steps
            >= self.seq_needed_num_from_last_micro_step
        )

    def balance(self):
        assert self.can_be_balanced()

        while self.seq_needed_num_from_last_micro_step > 0:
            popped_data = self._pop_from_previous_micro_steps()
            self.data[-1].fit_in(popped_data)

        for micro_step in self.data:
            micro_step.balance()

    def discard_last_micro_step(self) -> MicroStepGRPOData:
        assert len(self.data) > 0, "No micro steps to discard"
        return self.data.pop()

    def log(self, prefix: str = ""):
        logger.info(
            f"{prefix}GlobalStepGRPOData(micro_step_num={self.micro_step_num}, total_seq_num={self.total_seq_num}, token_budget={self.token_budget}, gpu_num={self.gpu_num})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)

    def verify(self):
        for micro_step in self.data:
            micro_step.verify()

        seen_pairs = set()
        for micro_step in self.data:
            for batch in micro_step.data:
                for item in batch.data:
                    pair = (item.prompt_idx, item.response_idx)
                    assert pair not in seen_pairs, (
                        f"Duplicate data found: prompt_idx={item.prompt_idx}, response_idx={item.response_idx}"
                    )
                    seen_pairs.add(pair)

        return True
