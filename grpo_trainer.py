import logging
import os
import importlib
import json

import torch
import torch.optim as optim
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from accelerate import Accelerator
import swanlab
import fire

from vllm_service.nccl_client import NCCLClient
from data_service.data_client import DataClient
from utils.metrics import MetricsManager
from utils.decorators import (
    track_time,
    track_metrics,
    track_memory,
    on_main_process,
    per_step,
    clear_and_log_metrics,
    catch_exception,
)
from config.utils import load_config, ConfigItem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    def __init__(self, config: ConfigItem):
        self.config = config
        self.accelerator = Accelerator()
        self.global_step = 0

        self.data_dir = os.path.join(self.config.training.save_dir, "data_training")
        if self.accelerator.is_main_process:
            os.makedirs(self.config.training.save_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)

        self._setup_clients()
        self._setup_model_and_optimizer()

    def _setup_clients(self):
        self.metrics = MetricsManager(accelerator=self.accelerator)
        self.data_client = DataClient(
            host="localhost", port=self.config.network.data_port
        )
        self.data_client.initialize()

        if self.accelerator.is_main_process:
            self.nccl_client = NCCLClient(
                host=self.config.network.vllm_host,
                server_port=self.config.network.vllm_port,
                nccl_port=self.config.network.nccl_port,
                nccl_device=self.accelerator.device,
                dp_size=self.config.vllm_server.data_parallel_size,
            )
            self.nccl_client.init_nccl()
            self.swanlab_run = swanlab.init(
                project=self.config.training.project_name,
                name=self.config.training.run_name,
            )

    def _setup_model_and_optimizer(self):
        model_module = importlib.import_module(self.config.model.impl_path)
        init_params = self.config.model.init_params.to_dict()
        init_params["device_map"] = self.accelerator.device
        processor_module = importlib.import_module(self.config.processor.impl_path)

        self.policy = model_module.TFModelImpl(init_params=init_params)
        self.processor = processor_module.TFProcessorImpl(
            init_params=self.config.processor.init_params.to_dict(),
            apply_chat_template_params=self.config.processor.apply_chat_template_params.to_dict(),
        )
        self.optimizer = optim.AdamW(
            self.policy.model.parameters(), lr=self.config.training.lr
        )
        self.policy.model, self.optimizer = self.accelerator.prepare(
            self.policy.model, self.optimizer
        )

    @track_time("get_state_dict")
    def _get_state_dict(self):
        state_dict = get_model_state_dict(
            self.policy.model,
            options=StateDictOptions(full_state_dict=True),
        )
        return state_dict

    @on_main_process
    @per_step("training.save_steps")
    @track_time("save_model")
    def _save_model(self, state_dict):
        save_path = os.path.join(
            self.config.training.save_dir, f"step_{self.global_step}"
        )
        logger.info(f"Saving model {save_path} at step {self.global_step}")

        self.policy.model.save_pretrained(save_path, state_dict=state_dict)
        self.processor.processor.save_pretrained(save_path)

    @on_main_process
    @per_step("training.eval_steps")
    @track_time("evaluation")
    def _evaluate_model(self):
        logger.info(f"Evaluating model at step {self.global_step}")
        metrics = self.data_client.run_evaluation()
        self.metrics.local_metrics.add_from_flat_dict(metrics)

    @on_main_process
    @track_time("weight_sync")
    def _update_vllm_weights(self, state_dict):
        total_params = sum(p.numel() for p in state_dict.values())
        logger.debug(f"Syncing weights to vLLM, total parameters: {total_params:,}")
        self.nccl_client.update_weights_nccl(state_dict)

    @catch_exception
    @track_time("waiting_for_data")
    def _fetch_data(self):
        rank_data, metrics = self.data_client.generate_data(
            self.global_step,
            rank=self.accelerator.process_index,
            update_step=True,
        )
        if metrics:
            self.metrics.local_metrics.add_from_flat_dict(metrics)
        return rank_data

    @track_metrics(names="entropy", prefix="Train", local_mode="sum", gather_mode="sum")
    def _compute_entropy(self, batch_data_per_gpu, inputs, outputs):
        batched_per_token_entropy = self.processor.get_batched_response_entropy(
            inputs, outputs
        )
        batch_data_per_gpu.set_data_fields(
            "per_token_entropy", batched_per_token_entropy
        )
        return batch_data_per_gpu.sum_with_denominator(
            batched_per_token_entropy,
            self.config.training.loss_type,
        )

    @track_metrics(names="kl", prefix="Train", local_mode="sum", gather_mode="sum")
    def _compute_kl_divergence(self, batch_data_per_gpu, batched_policy_logprobs):
        batched_ref_logprobs = batch_data_per_gpu.get_data_fields("ref_logprobs")
        batched_per_token_kl = []
        for pol_lp, ref_lp in zip(batched_policy_logprobs, batched_ref_logprobs):
            if ref_lp is None:
                batched_per_token_kl.append(None)
                continue

            per_token_kl = torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1
            per_token_kl = torch.clamp(per_token_kl, min=-10, max=10)
            batched_per_token_kl.append(per_token_kl)

        batch_data_per_gpu.set_data_fields("per_token_kl", batched_per_token_kl)
        return batch_data_per_gpu.sum_with_denominator(
            batched_per_token_kl,
            self.config.training.loss_type,
        )

    @track_time("forward_pass")
    @track_memory("forward_pass")
    def _forward_pass(self, batch_data_per_gpu):
        inputs = self.processor.prepare_inputs(
            batch_data_per_gpu.get_data_fields("conversation"),
            max_length=self.config.data_server.max_length,
        )
        outputs = self.policy.forward(inputs)

        # Compute policy logprobs
        batched_policy_logprobs = self.processor.get_batched_response_logprobs(
            inputs, outputs
        )
        batch_data_per_gpu.set_data_fields("pol_logprobs", batched_policy_logprobs)

        self._compute_entropy(batch_data_per_gpu, inputs, outputs)
        self._compute_kl_divergence(batch_data_per_gpu, batched_policy_logprobs)

    @track_metrics(names="loss", prefix="Train", local_mode="sum", gather_mode="avg")
    def _compute_loss(self, batch_data_per_gpu):
        batched_per_token_loss = []
        for pol_lp, per_token_kl, adv in zip(
            batch_data_per_gpu.get_data_fields("pol_logprobs"),
            batch_data_per_gpu.get_data_fields("per_token_kl"),
            batch_data_per_gpu.get_data_fields("advantage"),
        ):
            per_token_scaled_adv = torch.exp(pol_lp - pol_lp.detach()) * adv
            if per_token_kl is not None:
                per_token_loss = -(
                    per_token_scaled_adv - self.config.training.grpo_beta * per_token_kl
                )
            else:
                per_token_loss = -per_token_scaled_adv
            batched_per_token_loss.append(per_token_loss)

        batch_data_per_gpu.set_data_fields("per_token_loss", batched_per_token_loss)

        # Compute weights based on loss type
        total_loss = (
            batch_data_per_gpu.sum_with_denominator(
                batched_per_token_loss,
                self.config.training.loss_type,
            )
            * self.accelerator.num_processes
        )

        return total_loss

    @track_time("backward_pass")
    @track_memory("backward_pass")
    def _backward_pass(self, loss):
        self.accelerator.backward(loss)

    def _save_data(self, rank_data):
        data_file = os.path.join(
            self.data_dir,
            f"step_{self.global_step}_rank_{self.accelerator.process_index}.json",
        )
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump([batch.model_dump() for batch in rank_data], f, indent=4)

    @track_time("grad_clip")
    @track_metrics(
        names="grad_norm", prefix="Train", local_mode="avg", gather_mode="avg"
    )
    def _grad_clip(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(),
            max_norm=self.config.training.max_grad_norm,
        )
        grad_norm = grad_norm.full_tensor()
        return grad_norm.item()

    @track_time("optimizer_step")
    def _optimizer_step(self):
        self.optimizer.step()

    @track_time("micro_batch")
    def _process_micro_batch(self, batch_data_per_gpu):
        self._forward_pass(batch_data_per_gpu)
        loss = self._compute_loss(batch_data_per_gpu)
        self._backward_pass(loss)
        batch_data_per_gpu.detach()

    @clear_and_log_metrics
    @track_time("global_step")
    def _train_step(self):
        self.optimizer.zero_grad()

        rank_data = self._fetch_data()
        if rank_data:
            for batch in rank_data:
                self._process_micro_batch(batch)
            self._save_data(rank_data)
            del rank_data

            self._grad_clip()
            self._optimizer_step()

        state_dict = self._get_state_dict()
        self._update_vllm_weights(state_dict)
        self._save_model(state_dict)
        del state_dict

        self._evaluate_model()

    def _pre_training_preparation(self):
        state_dict = self._get_state_dict()
        self._update_vllm_weights(state_dict)
        del state_dict
        if self.accelerator.is_main_process:
            self.data_client.reset()

        # Ensure all ranks wait until reset is finished before requesting step data.
        self.accelerator.wait_for_everyone()

    def train(self):
        self._pre_training_preparation()

        for self.global_step in range(self.config.training.total_steps):
            self._train_step()

    def cleanup(self):
        if self.accelerator.is_main_process:
            self.nccl_client.close()
            self.swanlab_run.finish()
        self.data_client.close()
        self.accelerator.end_training()


def main(config_file: str):
    trainer = Trainer(load_config(config_file))
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    fire.Fire(main)
