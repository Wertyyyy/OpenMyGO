import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Coroutine
from dataclasses import dataclass, field

from fastapi import HTTPException

from data_service.typing.grpo_data import GRPOData, BatchedGRPOData
from data_service.typing.grouping import adaptive_grouping
from utils.metrics import LocalMetrics, MetricValue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class StepState:
    tasks: List[asyncio.Task] = field(default_factory=list)
    combined_data: Optional[List[List[GRPOData]]] = None
    step_metrics: Optional[LocalMetrics] = None
    grouped_data: Optional[Any] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def add_task(self, task_coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Register a task and invalidate cached aggregation/grouping state."""
        task = asyncio.create_task(task_coro)
        self.tasks.append(task)
        self.combined_data = None
        self.step_metrics = None
        self.grouped_data = None
        return task

    async def get_rank_data(
        self,
        *,
        step: int,
        rank: int,
        gpu_num: int,
        token_budget: int,
        max_micro_step_num: int,
        discard_clipped: bool,
        discard_aborted: bool,
    ) -> Tuple[List[BatchedGRPOData], Dict[str, MetricValue]]:
        if not self.tasks:
            raise HTTPException(status_code=404, detail="Data not found")

        async with self.lock:
            if self.combined_data is None or self.step_metrics is None:
                task_results = await asyncio.gather(*self.tasks, return_exceptions=True)

                combined_data: List[List[GRPOData]] = []
                combined_metrics = LocalMetrics()
                success_count = 0
                last_exception = None

                for result in task_results:
                    if isinstance(result, asyncio.CancelledError):
                        last_exception = result
                        logger.warning(
                            f"Data generation task for step {step} was cancelled"
                        )
                        continue

                    if isinstance(result, Exception):
                        last_exception = result
                        logger.error(f"Error fetching data for step {step}: {result}")
                        continue

                    filtered_data, step_metrics = result
                    combined_data.extend(filtered_data)
                    combined_metrics.add_from_flat_dict(step_metrics.to_flat_dict())
                    success_count += 1

                if success_count == 0:
                    if isinstance(last_exception, asyncio.CancelledError):
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"Data generation for step {step} was cancelled, "
                                "likely due to a server reset"
                            ),
                        )
                    if last_exception is not None:
                        raise last_exception
                    raise HTTPException(
                        status_code=500,
                        detail=f"No valid fetch result for step {step}",
                    )

                self.combined_data = combined_data
                self.step_metrics = combined_metrics

            if self.grouped_data is None:
                combined_data = list(self.combined_data)
                try:
                    self.grouped_data = adaptive_grouping(
                        combined_data,
                        gpu_num=gpu_num,
                        token_budget=token_budget,
                        max_micro_step_num=max_micro_step_num,
                        discard_clipped=discard_clipped,
                        discard_aborted=discard_aborted,
                        metrics=self.step_metrics,
                    )
                except Exception as e:
                    logger.error(f"Error during adaptive grouping for step {step}: {e}")
                    raise

            try:
                rank_data = []
                for micro_step_data in self.grouped_data.data:
                    rank_data.append(micro_step_data.data[rank])
            except Exception as e:
                logger.error(f"Error extracting rank {rank} data for step {step}: {e}")
                raise

            if rank == 0:
                metrics_dict = self.step_metrics.to_flat_dict()
            else:
                metrics_dict = {}

            return rank_data, metrics_dict
