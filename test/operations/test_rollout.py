import logging
import importlib
import copy

import fire

from config.utils import load_config
from vllm_service.vllm_client import VLLMClient
from test.test_data.convs import TEST_CONVERSATIONS_PURE_TEXT
from data_service.operations.rollout import RolloutOperation, RolloutInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rollout(config_file: str):
    logger.info(f"Loading config from {config_file}")
    config = load_config(config_file)

    # Initialize Processor
    processor_module = importlib.import_module(config.processor.impl_path)
    processor = processor_module.TFProcessorImpl(
        init_params=config.processor.init_params.to_dict(),
        apply_chat_template_params=config.processor.apply_chat_template_params.to_dict(),
    )
    logger.info("Processor initialized.")

    # Initialize VLLM Client
    logger.info(f"Initializing VLLM Client connecting to {config.network.vllm_host}:{config.network.vllm_port}")
    vllm_client = VLLMClient(
        host=config.network.vllm_host,
        server_port=config.network.vllm_port,
    )
    await vllm_client.initialize()
    
    try:
        # Prepare sampling params for testing
        sampling_params = config.data_server.generation_sampling_params.to_dict()
        if sampling_params.get("n", 1) == 1:
            sampling_params["n"] = 2  # default to 2 to test multiple rollouts

        rollout_op = RolloutOperation(
            vllm_client=vllm_client,
            processor=processor,
            sampling_params=sampling_params,
            max_length=config.data_server.max_length,
        )

        # Get Test Data
        conv_name = "single_turn"
        conversation = copy.deepcopy(TEST_CONVERSATIONS_PURE_TEXT[conv_name])
        conversation.messages = [m for m in conversation.messages if m.role != "assistant"]
        
        rollout_input = RolloutInput(
            conversation=conversation,
            solution="Dummy solution",
            prompt_idx=conv_name,
            label="Rollout-Test",
        )

        logger.info("Starting Rollout Operation...")
        rollout_stream = rollout_op(rollout_input)
        grpo_batch = [item async for item in rollout_stream]
        
        logger.info(f"Generated {len(grpo_batch)} responses.")
        for idx, item in enumerate(grpo_batch):
            logger.info(f"  [Rollout {idx}] : {item.conversation.messages[-1].content}")
            logger.info(f"  [Rollout {idx}] Response Token Length: {item.response_length}")

        assert len(grpo_batch) == sampling_params["n"], "Mismatch between generated responses and request 'n'"
        
        logger.info("Rollout test completed successfully!")

    finally:
        await vllm_client.close()

if __name__ == "__main__":
    fire.Fire(test_rollout)