import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
import contextlib
import traceback
from functools import partial
import os
import importlib
from collections import defaultdict

import torch
import fire
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm_service.vllm_client import VLLMClient
from tf_service.tf_client import TFClient
from config.utils import load_config, ConfigItem
from utils.metrics import LocalMetrics, MetricValue

from data_service.typing.grpo_data import GRPOData, BatchedGRPOData
from data_service.typing.message import Conversation
from data_service.data_utils import (
    save_data_generation_results,
    save_evaluation_results,
)
from data_service.metric_utils import (
    compute_rollout_metrics,
    compute_evaluation_metrics,
    compute_overall_evaluation_metrics,
)
from data_service.operations.rollout import RolloutOperation, RolloutInput
from data_service.operations.grpo import GRPOOperation
from data_service.operations.reward import RewardOperation
from data_service.operations.advantage import AdvantageOperation
from data_service.operations.reduce import GroupParamsOperation, GlobalParamsOperation
from data_service.step_state import StepState


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StepUpdateRequest(BaseModel):
    step: int


class DataGenerationRequest(BaseModel):
    step: int
    rank: int
    update_step: bool


class DataGenerationResponse(BaseModel):
    data: List[BatchedGRPOData]
    metrics: Optional[Dict[str, MetricValue]]


class EvaluationResponse(BaseModel):
    metrics: Dict[str, MetricValue]


class DataServer:
    def __init__(
        self,
        config: ConfigItem,
    ):
        self.config = config
        self.use_ref = self.config.training.grpo_beta > 0
        logger.info(f"Using reference model: {self.use_ref}")

        self.current_step = 0
        self.step_states: Dict[int, StepState] = defaultdict(StepState)

        self._setup_storage_dirs()
        self._initialize_components()

    def _setup_storage_dirs(self):
        self.data_dir = os.path.join(self.config.training.save_dir, "data_generation")
        self.eval_dir = os.path.join(self.config.training.save_dir, "evaluation")

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        logger.info(f"Data storage directory: {self.data_dir}")
        logger.info(f"Evaluation storage directory: {self.eval_dir}")

    def _initialize_components(self):
        # Load train dataset
        dataset_module = importlib.import_module(self.config.dataset.train.impl_path)
        self.dataset = dataset_module.TrainDatasetImpl(
            system_prompt_path=self.config.dataset.system_prompt_path,
            template_path=self.config.dataset.template_path,
            **self.config.dataset.train.init_params.to_dict(),
        )
        logger.info(
            f"Loaded train dataset implementation: {self.config.dataset.train.impl_path}"
        )

        # Load test datasets
        self.test_datasets = {}
        for test_name, test_config in self.config.dataset.test.items():
            test_dataset_module = importlib.import_module(test_config.impl_path)
            self.test_datasets[test_name] = test_dataset_module.TestDatasetImpl(
                system_prompt_path=self.config.dataset.system_prompt_path,
                template_path=self.config.dataset.template_path,
                **test_config.init_params.to_dict(),
            )
            logger.info(
                f"Loaded test dataset '{test_name}' implementation: {test_config.impl_path}"
            )

        # Load processor
        processor_module = importlib.import_module(self.config.processor.impl_path)
        self.processor = processor_module.TFProcessorImpl(
            init_params=self.config.processor.init_params.to_dict(),
            apply_chat_template_params=self.config.processor.apply_chat_template_params.to_dict(),
        )
        logger.info(
            f"Loaded processor implementation: {self.config.processor.impl_path}"
        )

        # Load reward functions
        self.reward_fns = {}
        for reward_name, reward_config in self.config.reward.items():
            try:
                impl_path = reward_config["impl_path"]
                init_params = reward_config["init_params"]

                reward_module = importlib.import_module(impl_path)
                # Instantiate the RewardImpl class with init_params
                reward_instance = reward_module.RewardImpl(**init_params)
                self.reward_fns[reward_name] = reward_instance
                logger.info(
                    f"Loaded reward function: {reward_name} from {impl_path} "
                    f"with params: {init_params}"
                )
            except Exception as e:
                logger.error(f"Failed to load reward function '{reward_name}': {e}")
                raise

        if not self.reward_fns:
            logger.error("No reward functions loaded successfully")
            raise RuntimeError("Failed to load any reward functions")

        self._filter_dataset()
        self._create_dataloader()

    def _filter_dataset(self):
        def filter_fn(example, collate_fn):
            conversations, *_ = collate_fn([example])
            seq_len = self.processor.get_seq_length(conversations[0])
            return seq_len <= self.config.data_server.max_prompt_length

        original_size = len(self.dataset.dataset)
        logger.info(f"Training dataset size before filtering: {original_size}")
        self.dataset.dataset = self.dataset.dataset.filter(
            partial(filter_fn, collate_fn=self.dataset.collate_fn),
            batched=False,
        )
        filtered_size = len(self.dataset.dataset)
        logger.info(f"Training dataset size after filtering: {filtered_size}")

    def _create_dataloader(self):
        def collate_wrapper(examples):
            return self.dataset.collate_fn(examples)

        dataloader = torch.utils.data.DataLoader(
            self.dataset.dataset,
            batch_size=self.config.data_server.global_batch_size,
            shuffle=True,
            collate_fn=collate_wrapper,
            drop_last=True,
        )
        self.dataloader = iter(dataloader)
        logger.info(
            f"DataLoader created with batch size: {self.config.data_server.global_batch_size}"
        )

    async def initialize_clients(self):
        self.policy_model_vllm_client = VLLMClient(
            host=self.config.network.vllm_host,
            server_port=self.config.network.vllm_port,
        )
        logger.info(
            f"Created VLLM client: {self.config.network.vllm_host}:{self.config.network.vllm_port}"
        )

        clients_to_init = []

        clients_to_init.append(self.policy_model_vllm_client.initialize())

        if self.use_ref:
            self.ref_model_tf_client = TFClient(
                host=self.config.network.tf_host, port=self.config.network.tf_port
            )
            logger.info(
                f"Created TF client: {self.config.network.tf_host}:{self.config.network.tf_port}"
            )
            clients_to_init.append(self.ref_model_tf_client.initialize())
        else:
            self.ref_model_tf_client = None

        await asyncio.gather(*clients_to_init)
        logger.info("Data service clients initialized successfully")

        self.rollout_op = RolloutOperation(
            vllm_client=self.policy_model_vllm_client,
            processor=self.processor,
            sampling_params=self.config.data_server.generation_sampling_params.to_dict(),
            max_length=self.config.data_server.max_length,
        )

        self.eval_rollout_op = RolloutOperation(
            vllm_client=self.policy_model_vllm_client,
            processor=self.processor,
            sampling_params=self.config.data_server.evaluation_sampling_params.to_dict(),
            max_length=self.config.data_server.max_length,
        )

        self.reward_op = RewardOperation(reward_fns=self.reward_fns)
        self.advantage_op = AdvantageOperation(use_std=self.config.training.use_std)
        self.group_params_op = GroupParamsOperation()
        self.grpo_op = GRPOOperation(
            rollout_op=self.rollout_op,
            reward_op=self.reward_op,
            group_params_op=self.group_params_op,
            advantage_op=self.advantage_op,
        )
        self.eval_grpo_op = GRPOOperation(
            rollout_op=self.eval_rollout_op,
            reward_op=self.reward_op,
            group_params_op=self.group_params_op,
            advantage_op=None,
        )
        self.global_params_op = GlobalParamsOperation(
            max_tokens=self.config.data_server.generation_sampling_params.max_tokens
        )

    async def close(self):
        if self.policy_model_vllm_client:
            try:
                await self.policy_model_vllm_client.close()
                logger.info("VLLM client closed successfully")
            except Exception as e:
                logger.error(f"Error closing VLLM client: {e}")

        if self.ref_model_tf_client:
            try:
                await self.ref_model_tf_client.close()
                logger.info("TF client closed successfully")
            except Exception as e:
                logger.error(f"Error closing TF client: {e}")

    async def reset(self):
        logger.info("Resetting data server state")

        # Cancel all pending fetching tasks
        for step, step_state in self.step_states.items():
            for task in step_state.tasks:
                if not task.done():
                    try:
                        task.cancel()
                        logger.debug(f"Cancelled fetching task for step {step}")
                    except Exception as e:
                        logger.error(f"Error cancelling task for step {step}: {e}")

        self.step_states.clear()
        self.current_step = 0
        self._create_dataloader()

        logger.info("Data server state reset completed")

    async def update_step(self, step: int):
        if step < self.current_step:
            logger.info(f"Updating step from {self.current_step} to {step}")
        self.current_step = step

        steps_to_remove = [s for s in self.step_states.keys() if s < step]
        for old_step in steps_to_remove:
            old_state = self.step_states.pop(old_step)
            for task in old_state.tasks:
                if not task.done():
                    task.cancel()
            logger.debug(f"Removed fetching task for old step {old_step}")

        for step_idx in range(
            step, step + self.config.data_server.pregenerate_steps + 1
        ):
            step_state = self.step_states[step_idx]
            if not step_state.tasks:
                logger.info(f"Fetching data for step {step_idx}")
                self.step_states[step_idx].add_task(self._generate_data(step_idx))
                await asyncio.sleep(1)
            else:
                logger.debug(
                    f"Step {step_idx} already has {len(step_state.tasks)} registered fetching task(s)"
                )

    async def get_generated_data(
        self, step: int, rank: int
    ) -> Tuple[List[BatchedGRPOData], Dict[str, MetricValue]]:
        step_state = self.step_states.get(step)
        if step_state is None or not step_state.tasks:
            raise HTTPException(status_code=404, detail="Data not found")

        return await step_state.get_rank_data(
            step=step,
            rank=rank,
            gpu_num=self.config.data_server.gpu_num,
            token_budget=self.config.data_server.token_budget,
            max_micro_step_num=self.config.data_server.max_micro_step_num,
            discard_clipped=self.config.data_server.discard.length,
            discard_aborted=self.config.data_server.discard.abort,
        )

    def _get_data_generation_batch(self) -> Tuple[List[Conversation], List[str]]:
        try:
            conversations, solutions, *_ = next(self.dataloader)
            return conversations, solutions
        except StopIteration:
            # Recreate dataloader when exhausted
            logger.info("DataLoader exhausted, recreating...")
            self._create_dataloader()
            conversations, solutions, *_ = next(self.dataloader)
            return conversations, solutions

    async def _generate_data(self, target_step: int):
        conversations, solutions = self._get_data_generation_batch()

        tasks = []
        for idx, (conversation, solution) in enumerate(zip(conversations, solutions)):
            rollout_input = RolloutInput(
                conversation=conversation,
                solution=solution,
                prompt_idx=idx,
                label="Rollout",
            )
            tasks.append(self.grpo_op(rollout_input))

        all_data: List[Optional[List[GRPOData]]] = await asyncio.gather(*tasks)
        filtered_data: List[List[GRPOData]] = [data for data in all_data if data]

        if not filtered_data:
            logger.error(
                f"All prompts failed for step {target_step}, no valid data generated"
            )
            raise RuntimeError(f"No valid data generated for step {target_step}")

        if len(filtered_data) < len(all_data):
            logger.warning(
                f"Step {target_step}: {len(all_data) - len(filtered_data)} out of {len(all_data)} "
                f"prompts failed and were skipped"
            )

        # Step 4: Calculate Global Response Params
        filtered_data = await self.global_params_op(filtered_data)

        # Step 5: Compute Rollout Metrics
        step_metrics = compute_rollout_metrics(filtered_data)

        save_data_generation_results(
            step=target_step,
            raw_data=filtered_data,
            metrics=step_metrics,
            output_dir=self.data_dir,
        )

        step_metrics.print_metrics(step=target_step)
        return filtered_data, step_metrics

    def _get_eval_conversations(self, dataset_name: str) -> List[Dict[str, Any]]:
        test_dataset = self.test_datasets[dataset_name]

        dataset_size = len(test_dataset.dataset)
        logger.info(
            f"Loading {dataset_size} samples from evaluation dataset '{dataset_name}'"
        )

        eval_samples: List[Dict[str, Any]] = []
        for sample_idx in range(dataset_size):
            example = test_dataset.dataset[sample_idx]
            conversations, solutions, _ = test_dataset.collate_fn([example])
            eval_samples.append(
                {
                    "conversation": conversations[0],
                    "solution": solutions[0],
                    "sample_idx": sample_idx,
                }
            )
        return eval_samples

    async def _eval_per_dataset(self, dataset_name: str) -> Dict[str, Any]:
        eval_samples = self._get_eval_conversations(dataset_name)

        if not eval_samples:
            raise RuntimeError(
                f"No valid samples for evaluation dataset '{dataset_name}'"
            )

        generation_tasks = [
            self.eval_grpo_op(
                RolloutInput(
                    conversation=sample["conversation"],
                    solution=sample["solution"],
                    prompt_idx=f"{dataset_name}_{sample['sample_idx']}",
                    label=f"Eval-{dataset_name}",
                )
            )
            for sample in eval_samples
        ]

        all_results: List[List[GRPOData]] = await asyncio.gather(*generation_tasks)
        all_results = [result for result in all_results if result]

        if not all_results:
            raise RuntimeError(f"All evaluation samples failed for '{dataset_name}'")

        dataset_metrics = compute_evaluation_metrics(dataset_name, all_results)

        save_evaluation_results(
            step=self.current_step,
            dataset_name=dataset_name,
            results=all_results,
            metrics=dataset_metrics,
            output_dir=self.eval_dir,
        )

        total_responses = sum(len(batch) for batch in all_results)
        logger.info(
            f"Evaluation completed for '{dataset_name}' with {len(all_results)} samples ({total_responses} responses)"
        )

        return {
            "dataset_name": dataset_name,
            "results": all_results,
            "metrics": dataset_metrics,
        }

    async def run_evaluation(self) -> Dict[str, MetricValue]:
        logger.info("Starting evaluation on all datasets")

        # Evaluate all datasets concurrently.
        logger.info(
            f"Running concurrent inference on {len(self.test_datasets)} datasets"
        )
        dataset_names = list(self.test_datasets.keys())
        dataset_tasks = [
            self._eval_per_dataset(dataset_name) for dataset_name in dataset_names
        ]
        dataset_results = await asyncio.gather(*dataset_tasks, return_exceptions=True)

        # Compute metrics for each dataset and combine
        combined_metrics = LocalMetrics()
        dataset_metrics_list = []  # Store metrics for overall computation
        failed_datasets = []

        for dataset_name, result in zip(dataset_names, dataset_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to evaluate dataset '{dataset_name}': {result}")
                failed_datasets.append(dataset_name)
                continue

            dataset_metrics = result["metrics"]
            for key, value in dataset_metrics.to_flat_dict().items():
                combined_metrics.add(key, value)
            dataset_metrics_list.append(result)

        if failed_datasets:
            logger.warning(f"Failed datasets: {failed_datasets}")

        if not dataset_metrics_list:
            logger.error("All evaluation datasets failed")
            raise RuntimeError("All evaluation datasets failed")

        # Compute overall weighted average metrics across all datasets
        if len(dataset_metrics_list) > 1:
            overall_metrics = compute_overall_evaluation_metrics(dataset_metrics_list)
            # Add overall metrics to combined metrics
            combined_metrics.add_from_flat_dict(overall_metrics.to_flat_dict())

        logger.info("Evaluation completed")
        combined_metrics.print_metrics()
        return combined_metrics.to_flat_dict()


def create_app(server: DataServer):
    app = FastAPI()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await server.initialize_clients()
        yield
        await server.close()

    app.router.lifespan_context = lifespan

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/update_step/")
    async def update_step(request: StepUpdateRequest):
        """Update current step and pregenerate data for future steps"""
        try:
            await server.update_step(request.step)
            return {"status": "success", "current_step": server.current_step}
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error updating step: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    @app.post("/reset/")
    async def reset():
        try:
            await server.reset()
            return {"status": "success", "message": "Server state reset successfully"}
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error resetting server: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    @app.post("/generate_data/", response_model=DataGenerationResponse)
    async def generate_data(request: DataGenerationRequest):
        try:
            if request.update_step:
                await server.update_step(request.step)

            data, metrics = await server.get_generated_data(request.step, request.rank)

            return DataGenerationResponse(data=data, metrics=metrics)
        except HTTPException:
            raise

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Error generating data for step {request.step}: {e}\n"
                f"Traceback:\n{tb_str}"
            )
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    @app.post("/run_evaluation/", response_model=EvaluationResponse)
    async def run_evaluation():
        try:
            metrics = await server.run_evaluation()
            return EvaluationResponse(metrics=metrics)

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error running evaluation: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    return app


def main(config_file: str):
    config = load_config(config_file)
    server = DataServer(config=config)
    app = create_app(server)

    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=config.network.data_port,
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(uvicorn_config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
