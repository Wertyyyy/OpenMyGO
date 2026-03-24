import logging
import importlib

import fire

from config.utils import load_config
from data_service.typing.grpo_data import GRPOData
from data_service.typing.message import Conversation
from data_service.operations.reward import RewardOperation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_reward(config_file: str):
    logger.info(f"Loading config from {config_file}")
    config = load_config(config_file)

    # Initialize Reward Functions
    reward_fns = {}
    for reward_name, reward_config in config.reward.items():
        try:
            impl_path = reward_config["impl_path"]
            init_params = reward_config["init_params"]
            reward_module = importlib.import_module(impl_path)
            reward_instance = reward_module.RewardImpl(**init_params)
            reward_fns[reward_name] = reward_instance
            logger.info(f"Loaded reward function: {reward_name}")
        except Exception as e:
            logger.error(f"Failed to load reward function '{reward_name}': {e}")

    if not reward_fns:
        logger.warning("No reward functions available to test. Exiting...")
        return
        
    reward_op = RewardOperation(reward_fns=reward_fns)

    # Create dummy GRPOData
    convo = Conversation(messages=[{"role": "user", "content": "Barbara Blackburn can type 212 words per minute.  Due to Carpal tunnel syndrome, Barbara cannot use her left hand for a while so her typing speed is now 40 words less per minute. If she is supposed to type a document with 3440 words, how many minutes will it take her to finish typing the document?"}])
    convo.add_message("assistant", "To find out how long it will take Barbara Blackburn to type a 3440-word document given her reduced typing speed, we can follow these steps:\n\n1. Determine Barbara's new typing speed.\n2. Use the new typing speed to calculate the total time required to type the 3440-word document.\n\nFirst, we know Barbara's original typing speed is 212 words per minute. Due to her Carpal tunnel syndrome, her typing speed is now 40 words per minute less. Therefore, her new typing speed is:\n\\[ 212 - 40 = 172 \\text{ words per minute} \\]\n\nNext, we need to find out how many minutes it will take her to type 3440 words at this reduced speed. We can do this by dividing the total number of words in the document by her new typing speed:\n\\[ \\text{Time (minutes)} = \\frac{\\text{Total words}}{\\text{New typing speed}} = \\frac{3440}{172} \\]\n\nTo perform the division:\n\\[ \\frac{3440}{172} = 20 \\]\n\nThus, it will take Barbara Blackburn \\(\\boxed{20}\\) minutes to finish typing the document.")
    
    dummy_data = GRPOData(
        prompt_idx="test",
        response_idx=0,
        conversation=convo,
        solution="20",
        stop_reason="stop",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[4, 5, 6],
    )

    logger.info("Starting Reward Operation...")
    processed_data = await reward_op(dummy_data)
    
    logger.info(f"Rewards dict: {processed_data.rewards}")
    logger.info(f"Reward reasons: {processed_data.reward_reasons}")
    logger.info(f"Total reward sum: {processed_data.reward_sum}")
    
    # Very basic assertion to ensure the keys are populated
    for r_name in reward_fns.keys():
        assert r_name in processed_data.rewards, f"{r_name} is missing from rewards dict"

    logger.info("Reward test completed successfully!")

if __name__ == "__main__":
    fire.Fire(test_reward)