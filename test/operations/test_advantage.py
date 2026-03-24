import asyncio
import logging

from data_service.typing.grpo_data import GRPOData
from data_service.typing.message import Conversation
from data_service.operations.advantage import AdvantageOperation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advantage():
    use_std = True
    advantage_op = AdvantageOperation(use_std=use_std)
    
    # Generate dummy data batch
    data_batch = []
    convo = Conversation(messages=[{"role": "user", "content": "Hello"}])
    
    # We create several responses with different rewards
    for i in range(4):
        item = GRPOData(
            prompt_idx="test",
            response_idx=i,
            conversation=convo,
            solution="4",
            stop_reason="stop",
            prompt_token_ids=[1, 2, 3],
            response_token_ids=[4, 5, 6],
            rewards={"reward_a": i * 1.0, "reward_b": 0.5},
        )
        data_batch.append(item)
        
    logger.info("Before Advantage Operation:")
    for d in data_batch:
        logger.info(f"  Item {d.response_idx} Reward sum: {d.reward_sum}")

    # Process Advantage
    logger.info("Starting Advantage Operation...")
    processed_batch = await advantage_op(data_batch)
    
    logger.info("After Advantage Operation:")
    adv_sum = 0
    for d in processed_batch:
        logger.info(f"  Item {d.response_idx} Advantage: {d.advantage}")
        assert d.advantage is not None, f"Advantage is missing for item {d.response_idx}"
        adv_sum += d.advantage
        
    logger.info(f"Sum of advantages: {adv_sum:.4f}")
    # Mathematical property: mean-centered advantages should sum to roughly zero
    assert abs(adv_sum) < 1e-4, "Advantages are not mean-centered correctly"
    
    logger.info("Advantage test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_advantage())