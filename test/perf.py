import logging
import sys
import importlib
import copy
import random

import torch
import torch.optim as optim
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from utils.metrics import MetricsManager
from utils.decorators import (
    track_time,
    track_memory,
    on_main_process,
    clear_and_log_metrics,
)
from config.utils import load_config
from data_service.typing.grouping import adaptive_grouping
from data_service.typing.grpo_data import GRPOData, GlobalStepGRPOData
from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PerformanceTester:
    def __init__(self, config_file: str):
        self.config = load_config(config_file)
        self.accelerator = Accelerator()
        self._setup_model()
        self.metrics = MetricsManager(accelerator=self.accelerator)
        self.global_step = 0

    def _setup_model(self):
        model_module = importlib.import_module(self.config.model.impl_path)
        init_params = self.config.model.init_params.to_dict()
        init_params["device_map"] = self.accelerator.device
        self.model = model_module.TFModelImpl(init_params=init_params)

        processor_module = importlib.import_module(self.config.processor.impl_path)
        self.processor = processor_module.TFProcessorImpl(
            init_params=self.config.processor.init_params.to_dict(),
            apply_chat_template_params=self.config.processor.apply_chat_template_params.to_dict(),
        )

        self.optimizer = optim.AdamW(
            self.model.model.parameters(), lr=self.config.training.lr * 100
        )

        self.model.model, self.optimizer = self.accelerator.prepare(
            self.model.model, self.optimizer
        )

    @on_main_process
    def _create_mock_grpo_data(self) -> GlobalStepGRPOData:
        response_length = self.config.data_server.generation_sampling_params.max_tokens
        num_prompts = (
            self.config.data_server.global_batch_size
            * self.config.data_server.generation_sampling_params.n
        )

        conversation = Conversation()
        conversation.add_message("user", "Mock question")
        conversation.add_message(
            "assistant",
            " ".join(
                ["apple"] * (response_length // 5)
            ),
        )
        prompt_token_ids, response_token_ids = (
            self.processor.get_prompt_response_token_ids(conversation)
        )

        all_data = []
        for prompt_idx in range(num_prompts):
            response_data = []
            for response_idx in range(3):
                grpo_data = GRPOData(
                    prompt_idx=prompt_idx,
                    response_idx=response_idx,
                    conversation=copy.deepcopy(conversation),
                    solution="",
                    stop_reason="stop",
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                    rewards={
                        "accuracy": random.random(),
                    },
                )
                response_data.append(grpo_data)
            all_data.append(response_data)

        global_step_grpo_data = adaptive_grouping(
            all_data,
            gpu_num=self.accelerator.num_processes,
            token_budget=self.config.data_server.token_budget,
            max_micro_step_num=self.config.data_server.max_micro_step_num,
            metrics=self.metrics,
        )
        return global_step_grpo_data

    @track_time("forward_pass")
    def _forward_pass(self, batch_data_per_gpu):
        conversations = batch_data_per_gpu.get_data_fields("conversation")
        inputs = self.processor.prepare_inputs(
            conversations, max_length=self.config.data_server.max_length
        )
        outputs = self.model.forward(inputs)
        return outputs

    def _compute_mock_loss(self, outputs):
        return torch.mean(torch.abs(outputs["logits"]))

    @track_time("backward_pass")
    @track_memory("backward_pass")
    def _backward_pass(self, loss):
        self.accelerator.backward(loss)

        # for name, param in self.model.model.named_parameters():
        #     if param.grad is not None:
        #         grad_abs_mean = param.grad.full_tensor().abs().mean().item()
        #         print(
        #             f"[Rank {self.accelerator.process_index}] Parameter '{name}' abs mean grad: {grad_abs_mean:.6f}"
        #         )
        #         break

    @track_time("optimizer_step")
    def _optimizer_step(self):
        self.optimizer.step()

    @track_time("gather_weights")
    def _gather_weights(self):
        state_dict = get_model_state_dict(
            model=self.model.model,
            options=StateDictOptions(full_state_dict=True),
        )
        return state_dict

    @clear_and_log_metrics
    @track_time("step")
    def step(self):
        self.optimizer.zero_grad()

        global_step_grpo_data = self._create_mock_grpo_data()
        global_step_grpo_data = broadcast_object_list(
            [global_step_grpo_data], from_process=0
        )[0]

        batch_data_per_gpu = global_step_grpo_data.data[0].data[
            self.accelerator.process_index
        ]
        outputs = self._forward_pass(batch_data_per_gpu)
        mock_loss = self._compute_mock_loss(outputs)
        self._backward_pass(mock_loss)
        self._optimizer_step()

    def train(self):
        self.accelerator.wait_for_everyone()
        for self.global_step in range(self.config.training.total_steps):
            self.step()


def main():
    tester = PerformanceTester(sys.argv[1])
    tester.train()


if __name__ == "__main__":
    main()
